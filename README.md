<p align="center">
<img src="assets/takenoko.png" alt="Takenoko Logo" width="300"/>
<h1 align="center">Takenoko</h1>
</p>

<i>An opinionated, perpetual WIP project aimed at hacking WanVideo 2.1(2)-T2V-(A)14B LoRA training.</i>
\
\
Initially based on the [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) codebase, it is intended as a playground for experimenting with new ideas and testing various training features, including those that might ultimately turn out to be useless. The config file structure may change at any time, and some non-functioning options might still be present.

<h3>📜 History</h3>

I originally planned to write a [WanVideo](https://huggingface.co/Wan-AI) trainer from scratch, but quickly realized I (for now) lack the knowledge to finish it <i>in a reasonable timeframe</i>. So I decided to start from [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) instead, since  it is simple, easy to modify, and fun to work with. That way, I could focus more on experimentation and learning.

I stripped out everything unrelated to Wan2.1-T2V-14B training and some other functions I don't need, refactored the code to my liking, added quality-of-life features I wanted (such as automatic TensorBoard logging, validation dataset support, MSE loss calculation, auto-resume training, experimental optimizers, more extensive logging, a single config file, a unified dialog-based training routine) and a lot of other experimental stuff. Some features I've added may be questionable in usefulness, and I haven't even tested them all yet.

<h3>📜 Installation (Windows)</h3>

1. Clone the repository.
2. Run `install.bat`.

<h3>📜 Quick Start</h3>

1. Create configuration file (you can copy sample config from `configs/examples` folder).
2. Place it into the `configs` directory.
3. Launch `run_trainer.bat` and follow the instructions.

<h3>📜 Docs</h3>

This is a personal project and probably won't have in the nearest future comprehensive docs (unless it somehow becomes popular). I've tried to provide detailed comments in the config template, but it can't cover eveything. As a workaround, I recommend using [repomix](https://github.com/yamadashy/repomix) to compress the whole repository into a single XML AI-readable file (will take around 400K tokens), then feeding it into the free Gemini 2.5 Pro with 1M context window (in [Google AI Studio](https://aistudio.google.com/)) and asking questions about various aspects of the project.

<h3>📜 Q&A</h3>

**❓ Does it only support Wan2.1-T2V-14B/Wan2.2-T2V-A14B LoRA training?**  
✔️ YES. 

**❓ Why not just fork and contribute to [musubi-tuner](https://github.com/kohya-ss/musubi-tuner)?**  
✔️ There are several reasons. First, [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) supports multiple models I don't use (HunyuanVideo, Wan2.2-I2V-A14B, Wan2.2-5B, Wan2.1-1.3B, Wan2.1-I2V-14B, FunControl, Framepack), which makes modifying the code harder. Second, I want to understand video model training in depth, and the best way is to dissect and rebuild an existing project "from the inside". Third, I often want to test ideas or implement things from papers that might break the main codebase, and PRs like that aren't likely to be merged quickly, so switching branches constantly would be annoying. Lastly, I wanted to use the cool logo you see above.

**❓ Where else does this project steal code from?**  
✔️ I draw <i>inspiration</i> and continue to learn from other projects, such as [AI Toolkit](https://github.com/ostris/ai-toolkit), [Diffusion Pipe](https://github.com/tdrussell/diffusion-pipe), [Simple Tuner](https://github.com/bghira/SimpleTuner), and multiple [arxiv](https://arxiv.org/) papers I can barely understand. I try to reference all sources at the top of code files - if I missed any, let me know.

<h3>📜 License</h3>

This project incorporates code from multiple open-source projects (mostly [musubi-tuner](https://github.com/kohya-ss/musubi-tuner)), which use Apache 2.0, MIT and AGPLv3 licenses. Since AGPLv3 is a strong copyleft license, including any AGPLv3 code likely means the entire project must be released under AGPLv3. This is my understanding based on public licensing information.

<h3>📜 Acknowledgments</h3>

This project would not be possible without [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) project by [kohya-ss](https://github.com/kohya-ss/). Although extensively refactored and adapted, the original work provided the foundation on which Takenoko was built. Thanks to [kohya-ss](https://github.com/kohya-ss/) for the awesome work.  
