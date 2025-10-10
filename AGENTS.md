# Repository Guidelines

## Project Structure & Module Organization
- `src/` is split by responsibility: `core/` contains the training orchestrators (`wan_network_trainer`, `training_core`, handlers), `dataset/` manages blueprint parsing and loaders, `criteria/` houses loss builders, `enhancements/` holds optional subsystems (alignment, self-correction, sliders, etc.), and `utils/` covers shared helpers.  
- `configs/` contains canonical TOML templates under `configs/examples/` and user configs at the root. Any new feature must be gated through `config_parser.py` and mirrored in the example template.  
- `tests/` is reserved for smoke/integration scenarios; treat it as the home for reproducible examples rather than unit micro-tests.  
- `tools/`, `extensions/`, `runpod/`, and `prebuilt/` provide auxiliary scripts, third-party hooks, and cached assets referenced by configuration.

## Architecture Overview & Flow
1. `config_parser.py` normalises TOML values into the argparse namespace and enforces defaults.  
2. `core/wan_network_trainer.py` prepares datasets, models, optimisers, and delegates the main loop to `core/training_core.py`.  
3. Enhancements plug in via factories (`enhancements/*_helper.py`) and must expose setup/teardown hooks so they can be prepared with `accelerator`.  
4. Loss composition happens in `criteria/training_loss.py`; optional modules should wire through here with clear gating to avoid accidental activation.

## Build, Test, and Development Commands
- `install.bat` — bootstrap Python environment on Windows.  
- `python -m compileall <path>` — fast syntax validation of touched modules (run on the package you changed, or `python -m compileall src`).  
- `run_trainer.bat --config configs/<name>.toml` — run the training loop locally; pair with low-step configs for smoke tests.  
- Profiling or log-inspection utilities live in `tools/`; read docstrings, many expect specific output directories.

## Coding Style & Naming Conventions
- Python 3.10+, four spaces, trailing commas on multi-line literals, and type hints for public signatures.  
- Files and symbols follow `snake_case.py`, `PascalCase` classes, `snake_case` functions/variables.  
- Use `common.logger.get_logger` for logging; prefer structured, levelled messages over print.  
- Keep comments high-signal. Long explanations belong in design docs (`*.md`) rather than inline.

## Configuration & Extension Guidelines
- New toggleable behaviour must include: config key in `config_parser.py`, default/documentation in `configs/examples/full_config_template.toml`, and guarded usage in code (`getattr(args, "flag", default)` pattern).  
- Provide concise, user-facing comments in the template so options remain self-explanatory.  
- When adding enhancements, place reusable logic under `enhancements/<feature>/` and ensure helpers expose `setup_hooks`, `remove_hooks`, and optional `get_<feature>_metrics` to mirror existing patterns.

## Testing Strategy
- At minimum, run `python -m compileall` on edited modules.  
- Add or update smoke scripts in `tests/` demonstrating new behaviour; keep them deterministic and small.  
- For long training changes, run a reduced-step config, capture key metrics/log excerpts, and reference them in your PR.  
- When touching dataset or config flows, validate `run_trainer.bat` with a dry-run config to ensure parsing still succeeds.

## Commit & Pull Request Guidelines
- Follow the observed convention `type(scope): short summary` (e.g., `feat(config): add grad clipping knobs`).  
- PRs must include: brief motivation, validation commands/log snippets, notes on config impact/backwards compatibility, and any relevant screenshots or metrics.  
- Reference related issues or TODOs; use checklists when multiple subtasks remain.  
- Squash commits where appropriate to keep history tidy.

## Security & Data Handling
- Never commit credentials, dataset URIs, or proprietary assets. Use local `.toml` overrides or environment variables for secrets.  
- Document any new configuration that expects sensitive values in README or template comments without exposing real data.  
- Clean up temporary artifacts (`output/`, `logs/`) before publishing branches.
