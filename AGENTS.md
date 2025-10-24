# Agent Playbook

## Purpose
This document is the operating manual for agents working in the Takenoko repository. It captures the repeatable workflow, validation expectations, approval rules, and the canonical reference material you should consult before making changes.

## Quick Start Checklist
1. Read the user goal carefully, capture ambiguities, and confirm constraints (sandbox mode, approvals, network access).  
2. Inspect relevant files using read-only commands (`rg`, `ls`, `Get-Content`), respecting existing uncommitted changes.  
3. Decide whether to create a plan: skip only for the simplest tasks; never submit a one-step plan.  
4. Implement changes using `apply_patch` for focused edits; avoid destructive Git commands.  
5. Validate with `python -m compileall <package>` for touched Python modules and any additional project-specific checks listed below.  
6. Craft the final response: lead with the changes made, reference files with `path:line`, suggest logical next steps, and avoid sharing raw command output.

## Workflow Details

### Intake & Triage
- Confirm environment context from the CLI banner (current directory, sandbox mode, approval policy).  
- Highlight missing information or risky assumptions to the user early; ask clarifying questions when behaviour is unclear.  
- Identify whether the task is a code change, review, documentation update, or exploratory investigation.

### Investigation
- Use `rg`/`rg --files`, `dir`, and targeted `Get-Content` to understand current state; prefer structured exploration over broad reads.  
- Cross-reference subsystem documentation (`src/<domain>/README.md`, `tools/`, `configs/examples/`) where available.  
- When you encounter unrelated uncommitted edits, avoid touching them and alert the user if they block progress.

### Planning & Execution
- Draft a plan with at least two steps for non-trivial work and update it as you complete each step.  
- Follow the editing constraints: stay ASCII, add comments only when they clarify complex logic, prefer `apply_patch` for manual edits, and never undo user changes.  
- Respect approval policy: request escalation with `with_escalated_permissions=true` and a one-line justification when a command needs to bypass the sandbox. Do not attempt destructive commands (`git reset --hard`, `git checkout --`) unless the user insists.  
- **For configuration additions** (CRITICAL TWO-STEP PATTERN):
  1. Add parameter to `configs/examples/full_config_template.toml` with:
     - Detailed comments explaining purpose, defaults, valid ranges/options, dependencies, and examples
     - Appropriate section grouping (e.g., training, optimizer, loss settings)
     - Clear naming convention (e.g., `enable_feature`, `feature_param_name`)
  2. Add parsing logic to `src/config_parser.py` in `create_args_from_config()`:
     - `args.param_name = config.get("param_name", default_value)`
     - Apply proper type conversion (bool, int, float, str, list, dict)
     - Include validation for value ranges, allowed options, dependencies
     - Add logging for important configuration changes
  3. Guard usage in code with `getattr(args, "flag", default)` for safety.
  - **NEVER add to only one location** - both template and parser are required for proper functionality.
  - Default values MUST match between template comments and parser.
  - Test with invalid values to ensure validation works correctly.  
- When integrating enhancements, expose `setup_hooks`, `remove_hooks`, and optional metric helpers inside `enhancements/<feature>/` just like existing modules.

### Validation
- Always run `python -m compileall <module_or_dir>` on modified Python packages (e.g., `python -m compileall src/core`).  
- If training loops, datasets, or configs change, run a dry run via `run_trainer.bat --config configs/<low-step-config>.toml` when feasible. Provide summaries of results instead of raw logs.  
- For documentation-only changes, proofread for accuracy, ensure links/paths exist, and verify markdown renders correctly (spot check headings, lists).  
- Capture validation commands and outcomes so they can be referenced in the final message.

### Reporting
- Responses should open with what changed and why. Reference files as `relative/path.py:42`.  
- Summarise validations performed (or explain why they were skipped).  
- Offer natural next steps, such as running integration tests or preparing a commit. Do not suggest irrelevant actions.  
- Keep the tone collaborative, concise, and self-contained; avoid large raw diffs or command transcripts.

## Tooling & Command Reference
- `install.bat`: Bootstrap Windows environments.  
- `python -m compileall src`: Syntax validation across the entire package; scope it to touched submodules when possible.  
- `run_trainer.bat --config configs/<name>.toml`: Execute the training loop; pair with low-step templates for smoke coverage.  
- Profiling and log utilities reside in `tools/`; read per-script docstrings before executing.

## Repository Reference
- `src/core/`: Training orchestrators (`wan_network_trainer.py`, `training_core.py`, handlers) plus accelerator integration.  
- `src/dataset/`: Blueprint parsing, dataset builders, and loader utilities.  
- `src/criteria/`: Loss composition (`training_loss.py`) and related factories; ensure optional modules are gated.  
- `src/enhancements/`: Plug-in subsystems (alignment, self-correction, sliders) exposing hook helpers.  
- `src/utils/`: Shared helpers—check here before re-implementing generic logic.  
- `assets/`: Static resources (figures, sample blueprints) referenced by documentation and tutorials.  
- `configs/`: Canonical templates live under `configs/examples/`; update these when introducing toggles and avoid touching user-level configs at the root unless asked.  
- `docs/`: Long-form design notes and how-to guides; link here from PRs when behaviour changes.  
- `extensions/`: Optional integrations and third-party hooks; follow existing patterns when adding new connectors.  
- `models/`: Pretrained checkpoints or conversion scripts; confirm licensing before adding new assets.  
- `output/` & `logs/`: Generated artifacts—leave them intact unless the user explicitly asks for cleanup.  
- `prebuilt/`: Cached weights, tokenizers, or artifacts shipped with the repo; avoid modifying without coordination.  
- `runpod/`: Deployment helpers and automation for remote execution environments.  
- `tests/`: Smoke and integration scenarios; prefer deterministic examples demonstrating new behaviour.  
- `tools/`: Maintenance scripts, profiling utilities, and log inspection helpers. Review docstrings before invocation.

## Coding Style & Naming Conventions
- Target Python 3.10+, four-space indentation, trailing commas on multi-line literals, and type hints for public APIs.  
- Use `snake_case.py` for files, `PascalCase` for classes, and `snake_case` for functions/variables.  
- Prefer `common.logger.get_logger` for structured logging; avoid `print`.  
- Keep comments high-signal; longer explanations belong in Markdown design docs.

## Testing & Release Expectations
- Augment or create smoke scenarios in `tests/` when behaviour changes. Keep them deterministic and documented.  
- For long training experiments, capture key metrics or logs and reference them in the final report or PR description.  
- Document any new config that consumes sensitive values; rely on environment variables or local overrides instead of committing secrets.  
- Clean up transient artifacts (`output/`, `logs/`) before publishing or handing off work.

## Common Code Flow Patterns

### 1. Configuration → Args → Component Initialization Flow
**Pattern**: TOML config → argparse.Namespace → Component managers

```
1. Load TOML config (config_parser.py::load_training_config)
2. Parse to argparse.Namespace (config_parser.py::create_args_from_config)
3. Initialize component managers with args
4. Components validate and process their specific flags
```

**Key Files**:
- `src/config_parser.py`: Central config parsing (1600+ lines)
- `src/takenoko.py`: UnifiedTrainer entry point
- `src/core/wan_network_trainer.py`: Main trainer orchestration

**Example**: Adding FP8 quantization support
```python
# 1. Add to full_config_template.toml
fp8_use_enhanced = true
fp8_quantization_mode = "block"

# 2. Parse in config_parser.py
args.fp8_use_enhanced = bool(config.get("fp8_use_enhanced", False))
args.fp8_quantization_mode = config.get("fp8_quantization_mode", "tensor")

# 3. Use in model_manager.py
if args.fp8_use_enhanced:
    apply_enhanced_fp8_quantization(model, args.fp8_quantization_mode)
```

### 2. Dataset Blueprint → DataLoader Pipeline
**Pattern**: User config → Blueprint → Dataset → DataLoader → Batch collation

```
1. Load user config from TOML (config_utils.py::load_user_config)
2. Sanitize config (ConfigSanitizer)
3. Generate blueprint (BlueprintGenerator::generate)
   - Separates train vs validation datasets
   - Applies fallback hierarchy: dataset config → general config → argparse → runtime params
4. Create dataset instances (generate_dataset_group_by_blueprint)
   - ImageDataset or VideoDataset based on blueprint
   - Apply bucketing, caching, frame extraction
5. Wrap in DataLoader with custom collator
6. Collator produces batched tensors with metadata
```

**Key Files**:
- `src/dataset/config_utils.py`: Blueprint generation (765 lines)
- `src/dataset/image_video_dataset.py`: Dataset implementations
- `src/dataset/buckets.py`: Resolution bucketing and batch management
- `src/utils/train_utils.py`: Collator class

**Example**: Adding video-specific dataset parameter
```python
# 1. Add to VideoDatasetParams in config_utils.py
@dataclass
class VideoDatasetParams(BaseDatasetParams):
    new_video_param: Optional[int] = None

# 2. Add to VIDEO_DATASET_DISTINCT_SCHEMA
VIDEO_DATASET_DISTINCT_SCHEMA = {
    "new_video_param": int,
    # ... existing fields
}

# 3. Use in VideoDataset class
def __init__(self, ..., new_video_param=None):
    self.new_video_param = new_video_param
```

### 3. Component Manager Architecture
**Pattern**: Separation of concerns via dedicated manager classes

**Core Managers** (all in `src/core/`):
- `ModelManager`: Model loading, FP8 quantization, device placement
- `OptimizerManager`: Optimizer creation, LR scheduling, gradient scaling
- `CheckpointManager`: Save/load checkpoints, state management, hooks
- `SamplingManager`: Inference sampling, validation generation
- `ControlSignalProcessor`: Control LoRA/ControlNet preprocessing

**Enhancement Helpers** (in `src/enhancements/`):
- `RepaHelper`: Representation alignment integration
- `SaraHelper`: Structural adversarial alignment
- `SliderTrainingManager`: Concept editing slider training
- `SelfCorrectionManager`: Self-correction cache management

**Training Helpers**:
- `TrainingCore`: Main training loop, loss computation, validation
- `VaeTrainingCore`: VAE-specific training logic
- `ValidationCore`: Validation metrics and perceptual analysis
- `AdaptiveTimestepManager`: Adaptive timestep sampling
- `FVDMManager`: Frame-wise velocity distribution matching

**Manager Initialization Pattern**:
```python
# In wan_network_trainer.py::train()
# 1. Create managers
self.model_manager = ModelManager()
self.optimizer_manager = OptimizerManager()
self.checkpoint_manager = CheckpointManager()

# 2. Handle model-specific setup
self.model_manager.handle_model_specific_args(args)

# 3. Load models via manager
transformer = self.model_manager.load_dit_model(args, accelerator)
vae = self.model_manager.load_vae_model(args, accelerator)

# 4. Create optimizer via manager
optimizer = self.optimizer_manager.create_optimizer(network, args)
lr_scheduler = self.optimizer_manager.create_lr_scheduler(optimizer, args)
```

### 4. Training Loop Flow with Handler Delegation
**Pattern**: Core loop delegates to specialized handlers for each concern

```
for epoch in epochs:
    for batch in dataloader:
        # 1. Prepare inputs (training_core.py)
        latents, text_embeds, timesteps = prepare_inputs(batch)
        
        # 2. Forward pass
        with accelerator.accumulate(network):
            model_pred = call_dit(transformer, network, latents, ...)
            loss = compute_loss(model_pred, target)
        
        # 3. Backward and optimize
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        
        # 4. Delegate to handlers (all in src/core/handlers/)
        handle_step_logging(...)           # logging_handler.py
        handle_step_validation(...)        # validation_handler.py
        handle_step_saving(...)            # saving_handler.py
        handle_training_sampling(...)      # sampling_handler.py
        handle_adaptive_timestep(...)      # adaptive_handler.py
        handle_self_correction_update(...) # self_correction_handler.py
        handle_vram_validation(...)        # vram_validation_handler.py
```

**Handler Responsibilities**:
- `logging_handler.py`: TensorBoard metrics, performance stats, component losses
- `validation_handler.py`: Validation dataset evaluation, perceptual metrics (LPIPS, SSIM, FVD)
- `saving_handler.py`: Checkpoint saving (model + state), cleanup old checkpoints
- `sampling_handler.py`: Training-time sampling for visual monitoring
- `adaptive_handler.py`: Adaptive timestep importance tracking
- `progress_bar_handler.py`: Enhanced tqdm with throughput, VRAM, EMA loss
- `metrics_utils.py`: Gradient norms, parameter stats, per-source losses
- `ema_utils.py`: Exponential moving average for loss smoothing

### 5. Enhancement Integration Pattern
**Pattern**: Optional enhancements via gated helpers with setup/remove hooks

**Standard Enhancement Interface**:
```python
class FeatureHelper:
    def __init__(self, diffusion_model, args):
        # Validate args.enable_feature
        # Initialize components
        pass
    
    def setup_hooks(self) -> None:
        # Attach forward hooks to model
        pass
    
    def remove_hooks(self) -> None:
        # Clean up hooks
        pass
    
    def compute_loss(self, inputs) -> torch.Tensor:
        # Return additional loss component
        pass
```

**Integration in Training Loop**:
```python
# 1. Initialize if enabled
repa_helper = None
if getattr(args, "enable_repa", False):
    repa_helper = RepaHelper(transformer, args)
    repa_helper.setup_hooks()

# 2. Compute additional loss
if repa_helper:
    repa_loss = repa_helper.get_repa_loss(clean_pixels)
    total_loss += args.repa_loss_lambda * repa_loss

# 3. Cleanup on exit
if repa_helper:
    repa_helper.remove_hooks()
```

**Examples in Codebase**:
- REPA: `enhancements/repa/repa_helper.py`
- SARA: `enhancements/sara/sara_helper.py`
- Slider: `enhancements/slider/slider_integration.py`
- Self-Correction: `enhancements/self_correction/setup.py`
- Temporal Consistency: `enhancements/temporal_consistency/training_integration.py`

### 6. Caching Operations Flow
**Pattern**: Unified caching for latents and text encoder outputs

```
1. Load dataset config (same blueprint generation as training)
2. Combine train + validation datasets
3. Load encoder model (VAE or T5)
4. Batch-process all items:
   - For latents: VAE encode frames → save .safetensors
   - For text: T5 encode captions → save .safetensors
5. Post-process: Remove orphaned cache files
6. Cleanup models and VRAM
```

**Key Files**:
- `src/caching/cache_latents.py`: VAE latent encoding
- `src/caching/cache_text_encoder_outputs.py`: T5 text embedding
- `src/caching/chunk_estimator.py`: Pre-training cache size estimation

**Usage Pattern**:
```python
# In takenoko.py::UnifiedTrainer
def cache_latents(self):
    # 1. Create blueprint (same as training)
    blueprint = blueprint_generator.generate(user_config, cache_args)
    
    # 2. Load VAE
    vae = WanVAE(vae_path, device, dtype)
    
    # 3. Encode and save
    encode_datasets(datasets, encode_callback, cache_args)
    
    # 4. Cleanup
    del vae
    torch.cuda.empty_cache()
```

### 7. Loss Computation Architecture
**Pattern**: Centralized loss computer with component aggregation

```
1. TrainingLossComputer (criteria/training_loss.py)
   - Base loss (MSE, L1, Huber, Pseudo-Huber, etc.)
   - Optional components:
     * Dispersive loss (diversity regularization)
     * Fourier loss (frequency domain)
     * DWT loss (wavelet domain)
     * Optical flow loss (temporal consistency)
     * Masked training loss (region-specific)

2. Enhancement losses (added separately):
   - REPA loss (representation alignment)
   - SARA loss (structural + adversarial)
   - Slider loss (concept editing)
   - Temporal consistency loss

3. Aggregation:
   total_loss = base_loss + Σ(λᵢ * component_lossᵢ)
```

**Adding New Loss Component**:
```python
# 1. Implement in src/criteria/new_loss.py
def compute_new_loss(pred, target, **kwargs):
    return loss_value

# 2. Add to TrainingLossComputer
class TrainingLossComputer:
    def compute_loss(self, ...):
        # Compute base loss
        base_loss = ...
        
        # Add new component
        new_loss = None
        if getattr(self.args, "enable_new_loss", False):
            new_loss = compute_new_loss(pred, target)
            total_loss += self.args.new_loss_weight * new_loss
        
        return LossComponents(base_loss, new_loss=new_loss)

# 3. Add config parameters (see pattern in section above)
```

## Critical Implementation Rules

### When Adding New Features:
1. **Always use the Manager/Helper pattern** for complex features
2. **Gate with enable_* flags** - never enable by default without user consent
3. **Provide setup_hooks() and remove_hooks()** for model modifications
4. **Log initialization clearly** - users need to know what's active
5. **Handle failures gracefully** - fallback or warn, don't crash training

### When Modifying Training Loop:
1. **Use handlers for cross-cutting concerns** - don't bloat training_core.py
2. **Maintain handler signatures** - breaking changes affect all callers
3. **Check accelerator.is_main_process** before logging/saving
4. **Use accelerator.sync_gradients** for periodic operations
5. **Respect global_step for checkpointing** - not iteration count

### When Adding Dataset Features:
1. **Update both ImageDatasetParams and VideoDatasetParams** if applicable
2. **Add to appropriate SCHEMA** in ConfigSanitizer
3. **Handle in BlueprintGenerator fallback chain**
4. **Test with bucket analyzer** before training
5. **Document in full_config_template.toml**

### Memory Management:
1. **Use SafeMemoryManager** for training-safe optimizations
2. **Call snapshot_gpu_memory()** at key points if trace_memory=true
3. **Use force_cuda_cleanup()** after major operations
4. **Offload to CPU when possible** (VAE caching, inactive models)
5. **Delete + empty_cache + gc.collect** for complete cleanup

Adhering to this playbook ensures consistent, auditable contributions and streamlined collaboration across the Takenoko project.
