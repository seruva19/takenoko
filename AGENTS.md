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
- For configuration additions, wire the flag through `config_parser.py`, document defaults in `configs/examples/full_config_template.toml`, and guard usage with `getattr(args, "flag", default)`.  
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

Adhering to this playbook ensures consistent, auditable contributions and streamlined collaboration across the Takenoko project.
