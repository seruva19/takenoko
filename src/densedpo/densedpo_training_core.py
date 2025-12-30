"""DenseDPO training core for segment-level preference optimization."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from scheduling.timestep_distribution import TimestepDistribution
from scheduling.timestep_utils import (
    get_noisy_model_input_and_timesteps,
    initialize_timestep_distribution,
)
from core.handlers.saving_handler import (
    handle_step_saving,
    handle_epoch_end_saving,
)

logger = logging.getLogger(__name__)


class DenseDPOTrainingCore:
    """DenseDPO training loop."""

    def __init__(
        self,
        *,
        densedpo_config,
        model_config: Dict[str, Any],
        accelerator,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        vae: Optional[torch.nn.Module],
        args,
        labeler=None,
    ) -> None:
        self.densedpo_config = densedpo_config
        self.model_config = model_config
        self.accelerator = accelerator
        self.transformer = transformer
        self.network = network
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.vae = vae
        self.args = args
        self.labeler = labeler
        self.timestep_distribution = TimestepDistribution()

        self.noise_scheduler = FlowMatchDiscreteScheduler(
            shift=getattr(args, "discrete_flow_shift", 1.0),
            reverse=True,
            solver="euler",
        )

    def run_training_loop(
        self,
        *,
        train_dataloader: Any,
        num_train_epochs: int,
        global_step: int,
        progress_bar: Optional[tqdm],
        save_model,
        remove_model,
        current_epoch=None,
        current_step=None,
        is_main_process: bool = False,
    ) -> int:
        logger.info("Starting DenseDPO training loop...")

        self.transformer.train()
        self.network.train()

        if progress_bar is None:
            progress_bar = tqdm(
                range(self.args.max_train_steps),
                initial=global_step,
                smoothing=0,
                disable=not self.accelerator.is_local_main_process,
                desc="steps",
                dynamic_ncols=True,
            )

        epoch_to_start = 0
        if global_step > 0:
            steps_per_epoch = len(train_dataloader)
            epoch_to_start = global_step // steps_per_epoch

        for epoch in range(epoch_to_start, num_train_epochs):
            if current_epoch is not None:
                current_epoch.value = epoch + 1
            if hasattr(self.network, "on_epoch_start"):
                try:
                    self.accelerator.unwrap_model(self.network).on_epoch_start(
                        self.transformer
                    )
                except Exception:
                    pass

            step_offset = 0
            if global_step > 0 and epoch == epoch_to_start:
                steps_per_epoch = len(train_dataloader)
                step_offset = global_step % steps_per_epoch

            for step, batch in enumerate(train_dataloader):
                if epoch == epoch_to_start and step < step_offset:
                    continue

                if current_step is not None:
                    current_step.value = global_step

                with self.accelerator.accumulate(self.network):
                    if hasattr(self.network, "on_step_start"):
                        try:
                            self.accelerator.unwrap_model(
                                self.network
                            ).on_step_start()
                        except Exception:
                            pass

                    loss, metrics = self._compute_dense_dpo_loss(batch)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.optimizer.step()
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                        global_step += 1
                        progress_bar.update(1)
                        if metrics:
                            progress_bar.set_postfix(metrics)

                        if (
                            getattr(self.args, "log_every_n_steps", None)
                            and global_step
                            % int(getattr(self.args, "log_every_n_steps", 1))
                            == 0
                        ):
                            try:
                                self.accelerator.log(
                                    {"densedpo/loss": float(loss.item())},
                                    step=global_step,
                                )
                            except Exception:
                                pass

                        should_save = False
                        save_every = getattr(
                            self.args, "save_every_n_steps", None
                        )
                        if save_every is not None and save_every > 0:
                            should_save = global_step % int(save_every) == 0

                        handle_step_saving(
                            should_saving=should_save,
                            accelerator=self.accelerator,
                            save_model=save_model,
                            remove_model=remove_model,
                            args=self.args,
                            network=self.network,
                            global_step=global_step,
                            epoch=epoch,
                        )

                        if global_step >= self.args.max_train_steps:
                            break

            handle_epoch_end_saving(
                args=self.args,
                epoch=epoch,
                num_train_epochs=num_train_epochs,
                is_main_process=is_main_process,
                save_model=save_model,
                remove_model=remove_model,
                accelerator=self.accelerator,
                network=self.network,
                global_step=global_step,
            )

            if global_step >= self.args.max_train_steps:
                break

        logger.info("DenseDPO training completed.")
        return global_step

    def _compute_dense_dpo_loss(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        latents = batch["latents"].to(
            device=self.accelerator.device, dtype=self.transformer.dtype
        )
        bsz = latents.shape[0]
        total_frames = latents.shape[2]
        segment_frames = int(self.densedpo_config.densedpo_segment_frames)
        num_segments = math.ceil(total_frames / segment_frames)

        prompts = self._get_prompts(batch, bsz)
        raw_context = batch["t5"]
        if torch.is_tensor(raw_context):
            context_list = [
                raw_context[i].to(
                    device=self.accelerator.device, dtype=self.transformer.dtype
                )
                for i in range(raw_context.shape[0])
            ]
        else:
            context_list = [
                t.to(
                    device=self.accelerator.device, dtype=self.transformer.dtype
                )
                for t in raw_context
            ]
        seq_len = self._infer_seq_len(latents)

        policy_latents, reference_latents = self._generate_pair(
            latents, context_list, seq_len
        )

        if self.densedpo_config.densedpo_label_source == "provided":
            preferences = self._get_provided_preferences(
                batch, bsz, num_segments
            )
            policy_rewards = None
            reference_rewards = None
        else:
            if self.labeler is None:
                raise RuntimeError(
                    "DenseDPO labeler missing for reward/VLM labeling."
                )
            if self.densedpo_config.densedpo_label_source == "vlm":
                policy_frames, reference_frames = self._build_segment_frames(
                    policy_latents, reference_latents, segment_frames
                )
                preferences, policy_rewards, reference_rewards = (
                    self.labeler.compute_preferences(
                        policy_frames=policy_frames,
                        reference_frames=reference_frames,
                    )
                )
            else:
                preferences, policy_rewards, reference_rewards = (
                    self.labeler.compute_preferences(
                        policy_latents=policy_latents,
                        reference_latents=reference_latents,
                        prompts=prompts,
                        segment_frames=segment_frames,
                    )
                )

        (
            logp_pi_policy,
            logp_ref_policy,
            logp_pi_reference,
            logp_ref_reference,
        ) = self._compute_logps(
            policy_latents=policy_latents,
            reference_latents=reference_latents,
            context_list=context_list,
            seq_len=seq_len,
        )

        prefs = preferences.to(self.accelerator.device, dtype=torch.float32)
        if prefs.shape != logp_pi_policy.shape:
            raise ValueError(
                "DenseDPO preferences shape mismatch: "
                f"expected {logp_pi_policy.shape}, got {prefs.shape}."
            )

        logit_diff = (
            (logp_pi_policy - logp_pi_reference)
            - (logp_ref_policy - logp_ref_reference)
        )
        sign = prefs * 2.0 - 1.0
        logits = self.densedpo_config.densedpo_beta * logit_diff * sign
        loss = -F.logsigmoid(logits).mean()

        metrics = {"loss": float(loss.item())}
        if policy_rewards is not None and reference_rewards is not None:
            metrics["reward_gap"] = float(
                (policy_rewards - reference_rewards).mean().item()
            )
        metrics["pref_rate"] = float(prefs.mean().item())
        return loss, metrics

    def _generate_pair(
        self,
        latents: torch.Tensor,
        context_list: List[torch.Tensor],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scheduler = FlowMatchDiscreteScheduler(
            shift=getattr(self.args, "discrete_flow_shift", 1.0),
            reverse=True,
            solver="euler",
        )
        scheduler.set_timesteps(
            self.densedpo_config.densedpo_num_inference_steps,
            device=self.accelerator.device,
        )
        start_index = self._infer_start_index(scheduler)

        noise = torch.randn_like(latents)
        eta = self.densedpo_config.densedpo_partial_noise_eta
        partial_latents = (1.0 - eta) * latents + eta * noise

        with torch.no_grad():
            policy_latents = self._denoise_from_partial(
                scheduler,
                partial_latents,
                context_list,
                seq_len,
                start_index,
                use_reference=False,
            )
            reference_latents = self._denoise_from_partial(
                scheduler,
                partial_latents,
                context_list,
                seq_len,
                start_index,
                use_reference=True,
            )
        return policy_latents, reference_latents

    def _denoise_from_partial(
        self,
        scheduler: FlowMatchDiscreteScheduler,
        latents: torch.Tensor,
        context_list: List[torch.Tensor],
        seq_len: int,
        start_index: int,
        *,
        use_reference: bool,
    ) -> torch.Tensor:
        scheduler.set_begin_index(start_index)
        scheduler._step_index = None
        current = latents
        for i in range(start_index, len(scheduler.timesteps)):
            timesteps = scheduler.timesteps[i].to(
                device=self.accelerator.device
            )
            timesteps = timesteps.expand(current.shape[0])
            latents_list = [current[j] for j in range(current.shape[0])]
            with self.accelerator.autocast():
                model_pred_list = self._run_model(
                    latents_list,
                    timesteps,
                    context_list,
                    seq_len,
                    use_reference=use_reference,
                )
            model_pred = torch.stack(model_pred_list, dim=0)
            current = scheduler.step(model_pred, timesteps, current).prev_sample
        return current

    def _compute_logps(
        self,
        *,
        policy_latents: torch.Tensor,
        reference_latents: torch.Tensor,
        context_list: List[torch.Tensor],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        initialize_timestep_distribution(
            self.args, self.timestep_distribution
        )
        presampled_uniform = torch.rand(
            policy_latents.shape[0],
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        noise = torch.randn_like(policy_latents)

        policy_inputs, timesteps, _ = get_noisy_model_input_and_timesteps(
            self.args,
            noise,
            policy_latents,
            self.noise_scheduler,
            self.accelerator.device,
            self.transformer.dtype,
            self.timestep_distribution,
            presampled_uniform=presampled_uniform,
        )
        reference_inputs, _, _ = get_noisy_model_input_and_timesteps(
            self.args,
            noise,
            reference_latents,
            self.noise_scheduler,
            self.accelerator.device,
            self.transformer.dtype,
            self.timestep_distribution,
            presampled_uniform=presampled_uniform,
        )

        with self.accelerator.autocast():
            policy_pred_list = self._run_model(
                [policy_inputs[j] for j in range(policy_inputs.shape[0])],
                timesteps,
                context_list,
                seq_len,
                use_reference=False,
            )
            reference_pred_list = self._run_model(
                [reference_inputs[j] for j in range(reference_inputs.shape[0])],
                timesteps,
                context_list,
                seq_len,
                use_reference=False,
            )
        policy_pred = torch.stack(policy_pred_list, dim=0)
        reference_pred = torch.stack(reference_pred_list, dim=0)

        with torch.no_grad():
            with self.accelerator.autocast():
                ref_policy_pred_list = self._run_model(
                    [policy_inputs[j] for j in range(policy_inputs.shape[0])],
                    timesteps,
                    context_list,
                    seq_len,
                    use_reference=True,
                )
                ref_reference_pred_list = self._run_model(
                    [
                        reference_inputs[j]
                        for j in range(reference_inputs.shape[0])
                    ],
                    timesteps,
                    context_list,
                    seq_len,
                    use_reference=True,
                )
        ref_policy_pred = torch.stack(ref_policy_pred_list, dim=0)
        ref_reference_pred = torch.stack(ref_reference_pred_list, dim=0)

        target_policy = noise - policy_latents
        target_reference = noise - reference_latents

        logp_pi_policy = self._segment_logps(policy_pred, target_policy)
        logp_pi_reference = self._segment_logps(
            reference_pred, target_reference
        )
        logp_ref_policy = self._segment_logps(
            ref_policy_pred, target_policy
        )
        logp_ref_reference = self._segment_logps(
            ref_reference_pred, target_reference
        )

        return (
            logp_pi_policy,
            logp_ref_policy,
            logp_pi_reference,
            logp_ref_reference,
        )

    def _segment_logps(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        error = (pred - target) ** 2
        per_frame = error.mean(dim=(1, 3, 4))
        return -self._segment_reduce(
            per_frame, self.densedpo_config.densedpo_segment_frames
        )

    def _segment_reduce(
        self, per_frame: torch.Tensor, segment_frames: int
    ) -> torch.Tensor:
        bsz, total_frames = per_frame.shape
        segments = []
        for start in range(0, total_frames, segment_frames):
            end = min(start + segment_frames, total_frames)
            segments.append(per_frame[:, start:end].mean(dim=1))
        return torch.stack(segments, dim=1).view(bsz, -1)

    def _build_segment_frames(
        self,
        policy_latents: torch.Tensor,
        reference_latents: torch.Tensor,
        segment_frames: int,
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        if self.vae is None:
            raise RuntimeError("DenseDPO VLM labeling requires a VAE.")
        bsz, _, total_frames, _, _ = policy_latents.shape
        policy_batches: List[List[torch.Tensor]] = []
        reference_batches: List[List[torch.Tensor]] = []
        for item_idx in range(bsz):
            policy_segments: List[torch.Tensor] = []
            reference_segments: List[torch.Tensor] = []
            for start in range(0, total_frames, segment_frames):
                end = min(start + segment_frames, total_frames)
                frame_indices = list(range(start, end))
                policy_segments.append(
                    self._decode_frames_single(
                        policy_latents[item_idx], frame_indices
                    )
                )
                reference_segments.append(
                    self._decode_frames_single(
                        reference_latents[item_idx], frame_indices
                    )
                )
            policy_batches.append(policy_segments)
            reference_batches.append(reference_segments)
        return policy_batches, reference_batches

    def _decode_frames_single(
        self, latents: torch.Tensor, frame_indices: List[int]
    ) -> torch.Tensor:
        with torch.no_grad():
            decoded_frames = []
            for frame_idx in frame_indices:
                frame_latents = latents[:, frame_idx, :, :]
                images_list = self.vae.decode([frame_latents])
                images = images_list[0].unsqueeze(0)
                images = (images + 1.0) / 2.0
                images = torch.clamp(images, 0.0, 1.0)
                decoded_frames.append(images[0])
            return torch.stack(decoded_frames, dim=0)

    def _run_model(
        self,
        latents_list: List[torch.Tensor],
        timesteps: torch.Tensor,
        context_list: List[torch.Tensor],
        seq_len: int,
        *,
        use_reference: bool,
    ):
        if use_reference:
            with self._reference_multiplier():
                return self.transformer(
                    latents_list,
                    t=timesteps,
                    context=context_list,
                    seq_len=seq_len,
                )
        return self.transformer(
            latents_list,
            t=timesteps,
            context=context_list,
            seq_len=seq_len,
        )

    def _reference_multiplier(self):
        class _RefContext:
            def __init__(self, network):
                self.network = network
                self.prev_multiplier = None

            def __enter__(self):
                if hasattr(self.network, "set_multiplier"):
                    self.prev_multiplier = getattr(
                        self.network, "multiplier", 1.0
                    )
                    self.network.set_multiplier(0.0)
                else:
                    raise RuntimeError(
                        "DenseDPO requires a LoRA network with set_multiplier."
                    )
                return self

            def __exit__(self, exc_type, exc, tb):
                if self.prev_multiplier is not None:
                    self.network.set_multiplier(self.prev_multiplier)
                return False

        return _RefContext(self.network)

    def _infer_start_index(
        self, scheduler: FlowMatchDiscreteScheduler
    ) -> int:
        eta = self.densedpo_config.densedpo_partial_noise_eta
        sigmas = scheduler.sigmas.to(self.accelerator.device)
        distances = torch.abs(sigmas - eta)
        start_index = int(torch.argmin(distances).item())
        return min(max(start_index, 0), len(scheduler.sigmas) - 2)

    def _get_prompts(self, batch: Dict[str, Any], bsz: int) -> List[str]:
        prompts = []
        item_infos = batch.get("item_info") or []
        if item_infos:
            for item in item_infos:
                prompts.append(getattr(item, "caption", "") or "")
        while len(prompts) < bsz:
            prompts.append("")
        return prompts

    def _get_provided_preferences(
        self, batch: Dict[str, Any], bsz: int, num_segments: int
    ) -> torch.Tensor:
        key = self.densedpo_config.densedpo_segment_preference_key
        prefs = batch.get(key)
        if prefs is None:
            raise ValueError(
                f"DenseDPO expected provided preferences in batch['{key}']."
            )
        prefs = prefs.to(self.accelerator.device, dtype=torch.float32)
        if prefs.dim() == 1:
            prefs = prefs.view(bsz, -1)
        if prefs.shape[0] != bsz or prefs.shape[1] != num_segments:
            raise ValueError(
                f"DenseDPO provided preferences shape mismatch: {prefs.shape}, "
                f"expected ({bsz}, {num_segments})."
            )
        prefs = prefs.clamp(0.0, 1.0)
        return prefs

    def _infer_seq_len(self, latents: torch.Tensor) -> int:
        lat_f, lat_h, lat_w = latents.shape[2:5]
        pt, ph, pw = self.model_config.patch_size
        return lat_f * lat_h * lat_w // (pt * ph * pw)
