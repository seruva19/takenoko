<p align="center">
<img src="assets/takenoko.png" alt="Takenoko Logo" width="300"/>
<h1 align="center">Takenoko</h1>
</p>

<i>An opinionated, perpetual WIP project aimed at hacking WanVideo 2.1(2)-T2V-(A)14B LoRA training.</i>
\
\
It is intended as a playground for experimenting with new ideas and testing various training features, including some that might ultimately turn out to be useless. The configuration file structure may change at any time, and some non-functioning options may still be present. <b>It only supports Wan2.1-T2V-14B and Wan2.2-T2V-A14B training.</b>

<h3>☄️ Disclaimer</h3>

This project would not have been possible without [musubi-tuner](https://github.com/kohya-ss/musubi-tuner). Although extensively refactored and reworked (to the point where upstream merge is no longer possible), the original project provided the foundation on which Takenoko was built. By reusing an existing and proven codebase, I was able to focus more on experimentation and learning instead of reinventing the wheel. Thanks to [kohya-ss](https://github.com/kohya-ss/) for the awesome work. 

<h3>☄️ Docs</h3>

Since this project is mostly aimed at personal use and is in a state of constant improvement (without guaranteeing backwards compatibility), it probably won't have comprehensive documentation in the near future (unless it somehow becomes popular, which I hope it does not). I've tried to provide detailed comments in the config template, but they can't cover everything. As a workaround, I recommend using [repomix](https://repomix.com/) to compress the entire repository into a single XML AI-readable file (will take around 1M tokens), then feeding it into the free [Grok 4 Fast](https://grok.com/) with 2M context window and asking questions about various aspects of the project.

<h3>☄️ Quick Start (Windows)</h3>

1. Clone the repository.
2. Run `install.bat`.
3. Create configuration file (you can copy sample config from `configs/examples` folder).
4. Place it into the `configs` directory.
5. Launch `run_trainer.bat` and follow the instructions.

<h3>☄️ License</h3>

This project borrows code from various sources, which use different types of licenses, mostly Apache 2.0, MIT, and AGPLv3. Since AGPLv3 is a strong copyleft license, including any AGPLv3 code likely means the entire project must be released under AGPLv3. This understanding is based on publicly available licensing information.

<h3>☄️ Acknowledgments</h3>

Takenoko draws inspiration from and incorporates code, ideas, and techniques from various open-source projects and publications. I thank the authors and maintainers for their contributions. Below is a list of all sources and papers (in no particular order). I have tried to reference all sources, but if I happen to miss any (or if more specific credits are warranted), please let me know.  
\
Keep in mind that work on some features is not yet complete due to time and hardware constraints. If a feature is not working or is not implemented exactly as in the original work, all responsibility lies with my implementation, not with the authors of the original code or paper.

| Source | Type | What was borrowed | Author(s) | License | Comment |
|--------|------|---------|---------|---------|---------|
| [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) | repo | - Original codebase | [kohya-ss](https://github.com/kohya-ss) | Apache 2.0 | |
| [blissful-tuner](https://github.com/Sarania/blissful-tuner) | repo | - Several optimization techniques | [Sarania](https://github.com/Sarania) | Apache 2.0 | |
| [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) | repo | - Pre-computed timestep distribution algorithm<br>- AdamW8bitKahan optimizer<br> - Automagic optimizer modifications | [tdrussell](https://github.com/tdrussell) | MIT |
| [WanTraining](https://github.com/spacepxl/WanTraining) | repo | - Control LoRA training<br>- DWT loss | [spacepxl](https://github.com/spacepxl) | Apache 2.0 | |
| [ai-toolkit](https://github.com/ostris/ai-toolkit) | repo | - Differential output preservation<br>- Adafactor optimizer<br>- Prodigy 8-bit optimizer<br>- Automagic optimizer<br>- EMA implementation<br>- Concept slider training<br>- Stepped loss | [ostris](https://github.com/ostris) | MIT |  |
| [musubi-tuner (pr)](https://github.com/kohya-ss/musubi-tuner/pull/63) | repo | - Initial implementation of validation datasets | [NSFW-API](https://github.com/NSFW-API) | Apache 2.0 | |
| [Timestep-Attention-and-other-shenanigans](https://github.com/Anzhc/Timestep-Attention-and-other-shenanigans) | repo | - Clustered MSE Loss<br>- EW loss | [Anzhc](https://github.com/Anzhc) | AGPL-3.0 | |
| [Diffuse and Disperse: Image Generation with Representation Regularization](https://arxiv.org/abs/2506.09027v1) | paper | - Dispersive loss | Runqian Wang, Kaiming He | CC BY 4.0 | |
| [DispLoss](https://github.com/raywang4/DispLoss) | repo | - Dispersive loss PyTorch implementation | [raywang4](https://github.com/raywang4) | MIT |  |
| [sd-scripts](https://github.com/kohya-ss/sd-scripts) | repo | - Regularization datasets<br>- LoRA-GGPO | [kohya-ss](https://github.com/kohya-ss) | Apache 2.0 | |
| [wan2.1-dilated-controlnet](https://github.com/TheDenk/wan2.1-dilated-controlnet) | repo | - ControlNET training | [TheDenk](https://github.com/TheDenk) | Apache 2.0 | |
| [T-LoRA](https://github.com/ControlGenAI/T-LoRA) | repo | - T-LoRA training | [ControlGenAI](https://github.com/ControlGenAI) | MIT | see also [paper](https://arxiv.org/abs/2507.05964) |
| [sd-scripts (fork)](https://github.com/hinablue/sd-scripts) | repo | - Fourier loss<br>- HinaAdaptive optimizer | [hinablue](https://github.com/hinablue) | Apache 2.0 | |
| [Muon](https://github.com/KellerJordan/Muon) | repo | - Muon optimizer | [KellerJordan](https://github.com/KellerJordan) | MIT | |
| [Sana](https://github.com/NVlabs/Sana) | repo | - CAME 8-bit optimizer | [NVlabs](https://github.com/NVlabs) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2410.10629) |
| [SimpleTuner](https://github.com/bghira/SimpleTuner) | repo | - Routed TREAD<br>- SOAP optimizer<br>- Masked training (spatial-first loss, area interpolation, proper normalization, auto mask generation)<br>- Advanced EMA features<br>- CREPA/LayerSync improvements | [bghira](https://github.com/bghira) | AGPL-3.0 | |
| [diffusion-pipe (pr)](https://github.com/Ada123-a/diffusion-pipe-TREAD) | repo | - Frame-based TREAD | [Ada123-a](https://github.com/Ada123-a) | MIT |  |
| [Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://arxiv.org/abs/2410.06940) | paper | -  Representational alignment loss, 3-layer MLP projection head, forward hook-based feature capture  | Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, Saining Xie | CC BY 4.0 | |
| [REPA](https://github.com/sihyun-yu/REPA) | repo | - Representation Alignment implementation | [sihyun-yu](https://github.com/sihyun-yu) | MIT |  |
| [dino](https://github.com/facebookresearch/dino) | repo | - VisionTransformer implementation | [facebookresearch](https://github.com/facebookresearch) | MIT | |
| [Sophia](https://github.com/Liuhong99/Sophia) | repo | - Sophia optimizer | [Liuhong99](https://github.com/Liuhong99) | MIT | see also [paper](https://arxiv.org/abs/2305.14342) |
| [Adaptive Non-Uniform Timestep Sampling for Diffusion Model Training](https://github.com/ku-dmlab/Adaptive-Timestep-Sampler) | repo | - Adaptive timestep sampling | [KU-DMLab](https://github.com/ku-dmlab) | MIT | see also [paper](https://arxiv.org/abs/2411.09998) |
| [Temporal Regularization Makes Your Video Generator Stronger](https://arxiv.org/abs/2503.15417) | paper | - Temporal regularization via perturbation | Harold Haodong Chen, Haojian Huang, Xianfeng Wu, Yexin Liu, Yajing Bai, Wen-Jie Shu, Harry Yang, Ser-Nam Lim | arXiv 1.0 | |
| [AR-Diffusion: Asynchronous Video Generation with Auto-Regressive Diffusion](https://arxiv.org/abs/2503.07418) | paper | - Frame-oriented Probability Propagation (FoPP) scheduler | Mingzhen Sun, Weining Wang, Gen Li, Jiawei Liu, Jiahui Sun, Wanquan Feng, Shanshan Lao, SiYu Zhou, Qian He, Jing Liu | arXiv 1.0 | |
| [Redefining Temporal Modeling in Video Diffusion: The Vectorized Timestep Approach](https://arxiv.org/abs/2410.03160) | paper | - Vectorized timestep scheduling | Yaofang Liu, Yumeng Ren, Xiaodong Cun, Aitor Artola, Yang Liu, Tieyong Zeng, Raymond H. Chan, Jean-michel Morel | arXiv 1.0 | |
| [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2506.08009) | paper | - Post-training autoregressive self-rollout method | Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, Eli Shechtman | CC BY-NC-SA 4.0 | |
| [Wan2.1-NABLA](https://github.com/gen-ai-team/Wan2.1-NABLA) | repo | - Dynamic sparse attention | [gen-ai-team](https://github.com/gen-ai-team) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2507.13546) |
| [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) | repo | - Reward LoRA training | [aigc-apps](https://github.com/aigc-apps) | Apache 2.0 | |
| [Fira](https://github.com/xichen-fy/Fira) | repo | - Fira optimizer | [xichen-fy](https://github.com/xichen-fy) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2410.01623) |
| [google-research](https://github.com/google-research/google-research) | repo | - Frechet Video Distance (FVD) implementation | [google-research](https://github.com/google-research) | Apache 2.0 | |
| [Mixture of Contexts for Long Video Generation](https://arxiv.org/abs/2508.21058) | paper | - Mixture of Contexts (MoC) sparse attention routing | Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, Maneesh Agrawala, Lu Jiang, Gordon Wetzstein | CC BY-SA 4.0 | |
| [SPHL-for-stable-diffusion](https://github.com/kabachuha/SPHL-for-stable-diffusion) | code | - Pseudo-Huber loss implementation | [kabachuha](https://github.com/kabachuha) |  | see also [paper](https://arxiv.org/abs/2403.16728) |
| [Context as Memory: Scene-Consistent Interactive Long Video Generation with Memory Retrieval](https://arxiv.org/abs/2506.03141) | paper | - Context-as-Memory integration | Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, Xihui Liu | CC BY 4.0 | |
| [SingLoRA](https://github.com/kyegomez/SingLoRA) | repo | - SingLoRA implementation | [kyegomez](https://github.com/kyegomez) | MIT | see also [paper](https://arxiv.org/abs/2507.05566) |
| [PEFT-SingLoRA](https://github.com/bghira/PEFT-SingLoRA) | repo | - Enhanced non-square matrix handling | [bghira](https://github.com/bghira) | BSD 2-clause |  |
| [musubi-tuner (pr)](https://github.com/kohya-ss/sd-scripts/pull/2010) | repo | - Latent quality analysis | [araleza](https://github.com/araleza) | Apache 2.0 | |
| [Contrastive Flow Matching](https://arxiv.org/abs/2506.05350v1) | paper | - Contrastive loss | George Stoica, Vivek Ramanujan, Xiang Fan, Ali Farhadi, Ranjay Krishna, Judy Hoffman | CC BY 4.0 | |
| [DeltaFM](https://github.com/gstoica27/DeltaFM) | repo | - Contrastive Flow Matching implementation (class-conditioned sampling, unconditional handling) | [gstoica27](https://github.com/gstoica27) | MIT |  |
| [OneTrainer](https://github.com/Nerogar/OneTrainer) | repo | - Masked training (prior preservation, unmasked weight, random mask removal) | [Nerogar](https://github.com/Nerogar) | AGPL-3.0 | |
| [Ouroboros-Diffusion: Exploring Consistent Content Generation in Tuning-free Long Video Diffusion](https://arxiv.org/abs/2501.09019) | paper | - Frequency-domain temporal consistency | Jingyuan Chen, Fuchen Long, Jie An, Zhaofan Qiu, Ting Yao, Jiebo Luo, Tao Mei | CC BY-SA 4.0 | |
| [mmgp](https://github.com/deepbeepmeep/mmgp) | repo | - Memory-mapped safetensors loading | [deepbeepmeep](https://github.com/deepbeepmeep) | GNU GPL |  |
| [attention-map-diffusers](https://github.com/wooyeolbaek/attention-map-diffusers) | repo | - Cross-attention map visualization | [wooyeolbaek](https://github.com/wooyeolbaek) | MIT |  |
| [musubi-tuner (fork)](https://github.com/betterftr/musubi-tuner) | repo | - Full model fine-tuning<br>- Row-based TREAD | [betterftr](https://github.com/betterftr) | Apache 2.0 | |
| [stochastic_round_cuda](https://github.com/ethansmith2000/stochastic_round_cuda) | repo | - Stochastic rounding CUDA implementation | [ethansmith2000](https://github.com/ethansmith2000) | MIT | |
| [simplevae](https://huggingface.co/AiArtLab/simplevae) | repo | - VAE training enhancements | [AiArtLab](https://huggingface.co/AiArtLab) |  | |
| [RamTorch](https://github.com/lodestone-rock/RamTorch) | repo | - RamTorch CPU-bouncing linear layers | [lodestone-rock](https://github.com/lodestone-rock) | Apache 2.0 | |
| [Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference](https://github.com/Tencent-Hunyuan/SRPO/) | repo | - SRPO preference optimization | [Tencent-Hunyuan](https://github.com/Tencent-Hunyuan) | SRPO Non-Commercial License | see also [paper](https://arxiv.org/abs/2509.06942) |
| [SARA: Structural and Adversarial Representation Alignment for Training-efficient Diffusion Models](https://arxiv.org/abs/2503.08253v1) | paper | - Autocorrelation matrix alignment<br>- Adversarial distribution alignment<br>- Multi-level hierarchical representation loss | Hesen Chen, Junyan Wang, Zhiyu Tan, Hao Li | CC BY 4.0 | |
| [Scion](https://github.com/LIONS-EPFL/scion) | repo | - Scion optimizer | [LIONS-EPFL](https://github.com/LIONS-EPFL) | MIT | see also [paper](https://arxiv.org/abs/2502.07529)  |
| [EqM](https://github.com/raywang4/EqM) | repo | - Equilibrium matching adaptation | [raywang4](https://github.com/raywang4) | MIT | see also [paper](https://arxiv.org/abs/2510.02300)  |
| [NorMuon](https://github.com/CoffeeVampir3/NorMuon) | repo | - Neuron-wise Normalized Muon implementation | [CoffeeVampir3](https://github.com/CoffeeVampir3) | MIT | |
| [TiM](https://github.com/WZDTHU/TiM) | repo | - Transition training objective (paired timesteps, transports, weighting, EMA) | [WZDTHU](https://github.com/WZDTHU) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2509.04394) |
| [rcm](https://github.com/NVlabs/rcm) | repo | - rCM distillation algorithm reference | [NVlabs](https://github.com/NVlabs) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2510.08431) |
| [Aozora_SDXL_Training](https://github.com/Hysocs/Aozora_SDXL_Training) | repo | - Raven optimizer | [Hysocs](https://github.com/Hysocs) |  | |
| [Sprint: Sparse-Dense Residual Fusion for Efficient Diffusion Transformers](https://arxiv.org/abs/2510.21986v1) | paper | - Sparse-dense residual fusion with token dropping<br>- Path-drop learning with token regularization<br>- Two-stage training scheduler | Dogyun Park, Moayed Haji-Ali, Yanyu Li, Willi Menapace, Sergey Tulyakov, Hyunwoo J. Kim, Aliaksandr Siarohin, Anil Kag | CC BY 4.0 | |
| [AdaMuon](https://github.com/Chongjie-Si/AdaMuon) | repo | - Adaptive Muon optimizer implementation | [Chongjie-Si](https://github.com/Chongjie-Si) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2507.11005) |
| [Cross-Frame Representation Alignment for Fine-Tuning Video Diffusion Models](https://arxiv.org/abs/2506.09229) | paper | - Cross-frame representation alignment | Sungwon Hwang, Hyojin Jang, Kinam Kim, Minho Park, Jaegul Choo | CC BY 4.0 | |
| [LayerSync: Self-aligning Intermediate Layers](https://github.com/vita-epfl/LayerSync) | repo | - Inter-layer alignment loss | [vita-epfl](https://github.com/vita-epfl) | MIT | see also [paper](https://arxiv.org/abs/2510.12581) |
| [HyperLoRA](https://github.com/bytedance/ComfyUI-HyperLoRA) | repo | - HyperLoRA concept | [bytedance](https://github.com/bytedance) | GPL-3.0 | see also [paper](https://arxiv.org/abs/2503.16944) |
| [Qwen-Image-i2L](https://huggingface.co/DiffSynth-Studio/Qwen-Image-i2L) | repo | - Trainable single-pass LoRA weight prediction hypernetwork concept | [DiffSynth-Studio](https://huggingface.co/DiffSynth-Studio) | Apache 2.0 | see also [article](https://huggingface.co/blog/kelseye/qwen-image-i2l) | |
| [iREPA](https://github.com/End2End-Diffusion/iREPA) | repo | - Convolutional projector for spatial preservation<br>- Spatial z-score normalization for sharper alignment | [End2End-Diffusion](https://github.com/End2End-Diffusion) | MIT | see also [paper](https://arxiv.org/abs/2512.10794) |
| [SpeedrunDiT](https://github.com/SwayStar123/SpeedrunDiT) | repo | - Dim-aware timestep shift<br>- Cross-batch CFM regularizer<br>- Sprint uncond-only path drop for sampling  | [SwayStar123](https://github.com/SwayStar123) | MIT |  |
| [Improved Variational Online Newton (IVON)](https://github.com/team-approx-bayes/ivon) | repo | - IVON implementation | [team-approx-bayes](https://github.com/team-approx-bayes) | GPL-3.0 | with code from [PR](https://github.com/team-approx-bayes/ivon/pull/7) by [rockerBOO](https://github.com/rockerBOO) |
| [MemFlow](https://github.com/KlingTeam/MemFlow) | repo | - Memory bank<br>- Sparse memory activation guidance | [KlingTeam](https://github.com/KlingTeam) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2512.14699) |
| [HASTE](https://github.com/NUS-HPC-AI-Lab/HASTE) | repo | - Holistic alignment loss<br>- Semantic anchor feature projections<br>- Attention alignment with teacher offset<br>- Stage‑wise termination | [NUS-HPC-AI-Lab](https://github.com/NUS-HPC-AI-Lab) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2505.16792) |
| [sd-scripts (pr)](https://github.com/kohya-ss/sd-scripts/pull/2221) | repo | - CDC-FM flow matching | [rockerBOO](https://github.com/rockerBOO) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2510.05930) |
| [GaLore](https://github.com/jiaweizzhao/GaLore) | repo | - GaLore optimizer | [jiaweizzhao](https://github.com/jiaweizzhao) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2403.03507) |
| [REG](https://github.com/Martinser/REG) | repo | - Class‑token entanglement<br>- Class‑token denoising loss<br>- Alignment loss to encoder features| [Martinser](https://github.com/Martinser) | MIT | see also [paper](https://arxiv.org/abs/2507.01467v2) |
| [Q-GaLore](https://github.com/VITA-Group/Q-GaLore) | repo | - Q-GaLore optimizer | [VITA-Group](https://github.com/VITA-Group) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2407.08296) |
| [SemanticGen: Video Generation in Semantic Space](https://arxiv.org/abs/2512.20619) | paper | - Semantic token conditioning<br>- Feature‑representation cross‑alignment loss | Jianhong Bai, Xiaoshi Wu, Xintao Wang, Xiao Fu, Yuanxing Zhang, Qinghe Wang, Xiaoyu Shi, Menghan Xia, Zuozhu Liu, Haoji Hu, Pengfei Wan, Kun Gai |  | |
| [transformers (pr)](https://github.com/huggingface/transformers/pull/31936) | repo | - Implementation of Q-GaLore optimizer | [SunMarc](https://github.com/SunMarc) | Apache 2.0 |  |
| [Glance](https://github.com/CSU-JPG/Glance) | repo | - Fixed-timestep distillation mode | [CSU-JPG](https://github.com/CSU-JPG) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2512.02899) |
| [Stable-Video-Infinity](https://github.com/vita-epfl/Stable-Video-Infinity) | repo | - Error‑recycling fine‑tuning<br>- Timestep‑grid replay buffers<br>- Buffer replacement strategies<br>- Warmup distributed buffer fill<br>- Probabilistic error injection and modulation<br>- Anchor‑conditioned motion replay<br>- Sequence‑aware batching for replay continuity | [vita-epfl](https://github.com/vita-epfl) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2510.09212) |
| [EquiVDM: Equivariant Video Diffusion Models with Temporally Consistent Noise](https://arxiv.org/abs/2504.09789v1) | paper | - Temporally consistent noise with flow caching | Chao Liu, Arash Vahdat  | CC BY 4.0 | |
| [catlvdm](https://github.com/chikap421/catlvdm) | repo | - BCNI/SACN corruption for T5 conditioning<br>- Structured corruption robustness boost <br>- Mask‑aware embedding noise injection | [chikap421](https://github.com/chikap421) | MIT | see also [paper](https://arxiv.org/abs/2505.21545) |
| [TPDiff: Temporal Pyramid Video Diffusion Model](https://arxiv.org/abs/2503.09566) | paper | - Temporal pyramid bounded sampling<br>- Stage‑wise temporal resampling<br>- Stage‑specific scheduler‑aware gamma/sigma | Lingmin Ran, Mike Zheng Shou | CC BY 4.0 | |
| [relora](https://github.com/Guitaricet/relora) | repo | - ReLoRA pipeline | [Guitaricet](https://github.com/Guitaricet) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2307.05695) |
| [DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models](https://arxiv.org/abs/2506.03517) | paper | - DenseDPO training method | Ziyi Wu, Anil Kag, Ivan Skorokhodov, Willi Menapace, Ashkan Mirzaei, Igor Gilitschenski, Sergey Tulyakov, Aliaksandr Siarohin | CC BY 4.0 | |
| [Blockwise-Flow-Matching](https://github.com/mlvlab/Blockwise-Flow-Matching) | repo | - Blockwise timestep segment objective<br>- SemFeat alignment conditioning<br>- SemFeat time-embedding injection<br>- FRN loss| [mlvlab](https://github.com/mlvlab) |  | see also [paper](https://arxiv.org/abs/2510.21167) |
| [MuonClip](https://github.com/kyegomez/MuonClip) | repo | - MuonClip | [kyegomez](https://github.com/kyegomez) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2507.20534) |
| [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) | paper | - Multi-path residual stream with learnable residual mixing matrix<br>- Doubly-stochastic manifold constraint<br>- Identity-mapping preservation across depth<br>- Sinkhorn-Knopp normalization enforcing constraint<br>- Norm-preserving cross-stream residual propagation | Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, Wenfeng Liang | arXiv 1.0 |  |
| [manifolds](https://github.com/thinking-machines-lab/manifolds) | repo | - Manifold Muon integration | [thinking-machines-lab](https://github.com/thinking-machines-lab) | MIT | see also [blogpost](https://thinkingmachines.ai/blog/modular-manifolds/) |
| [LoRA meets Riemannion: Muon Optimizer for Parametrization-independent Low-Rank Adapters](https://arxiv.org/abs/2507.12142) | paper | - Riemannion fixed‑rank optimizer<br>- Manifold momentum/transport<br>- Manifold‑aware LoRA tangent projection and retraction<br>- One‑step gradient locally optimal initialization  | Vladimir Bogachev, Vladimir Aletov, Alexander Molozhavenko, Denis Bobkov, Vera Soboleva, Aibek Alanov, Maxim Rakhuba | arXiv 1.0 |  |
| [pico-relora](https://github.com/Yu-val-weiss/pico-relora) | repo | - Optimizer reset via random pruning<br>- Jagged cosine scheduler | [Yu-val-weiss](https://github.com/Yu-val-weiss) | Apache 2.0 | see also [paper](https://arxiv.org/abs/2509.12960) |
| [Physics-Guided Motion Loss for Video Generation Model](https://arxiv.org/abs/2506.02244) | paper | - Physics-guided motion loss | Bowen Xue, Giuseppe Claudio Guarnera, Shuang Zhao, Zahra Montazeri | arXiv 1.0 |  |
| [optimizers](https://github.com/NoteDance/optimizers) | repo | - Original implementation of Kron, Conda, VSGD, RangerVA and NvNovoGrad optimizers | [NoteDance](https://github.com/NoteDance) | Apache 2.0 |  |
| [clora](https://github.com/gemlab-vt/clora) | repo | - Cross-attention capture<br>- Token-focused attention<br>- Spatial attention masking<br>- Contrastive attention separation | [gemlab-vt](https://github.com/gemlab-vt) | MIT | see also [paper](https://arxiv.org/abs/2403.19776) |
