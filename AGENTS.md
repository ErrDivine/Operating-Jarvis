# Repository Guidelines
## Project Structure & Module Organization
The repository centers on `agent.py`, where you implement `CustomAgent`. Keep new logic and helper functions inside this module; evaluation harness imports only this file. Use `demo_agent.py` for reference implementations (`DirectAgent`, `HierarchicalAgent`). `bash_run.py` batches interactions with datasets under `data/`, writing `results.json`. `prompts/tools_v0.json` defines callable tool schemas; keep it in sync with your agent's prompt. Pretrained weights sit in `models/`, downloaded via `models/download.sh`. Ad hoc smoke tests live in `casual_tests/`.

## Build, Test, and Development Commands
- `conda env create -f environment.yaml` installs the Python 3.10 toolchain.
- `conda activate huawei_agent` switches into the competition environment.
- `bash models/download.sh` fetches the Qwen checkpoint expected by the demos.
- `python bash_run.py` runs the selected agent over `data/单轮-冒烟测试集.jsonl` and saves responses for inspection.
- `python casual_tests/qwen_sample_code.py` verifies the language model can tokenize and generate.

## Coding Style & Naming Conventions
Follow standard PEP 8: four-space indentation, `snake_case` for functions, and `CamelCase` for classes. Keep public method docstrings concise and prefer pure functions for tool selection logic. Do not change the signature of `CustomAgent.__init__` or `CustomAgent.run`; attach helpers inside the class or as private module functions. Use f-strings and type hints (PEP 484) when adding new utilities.

## Testing Guidelines
Target deterministic outputs by seeding any random components. Extend `bash_run.py` locally if you need multiple datasets, but restore defaults before submitting. Uploaders expect only `agent.py` and `models/`; confirm `results.json` looks correct before delivery. For quick regression coverage, wrap tool-calling logic in unit-testable helpers and hit them in a separate harness under `casual_tests/`.

## Commit & Pull Request Guidelines
Git history currently uses concise, capitalized summaries (`Initial commit.`). Continue with imperative, <=50-character subject lines (e.g., `Add hierarchical tool planner`). Reference related tasks in the body, describe expected impact, and note datasets used for validation. Pull requests should include a short scenario list, reproduction commands, and mention any deviations from the submission packaging rules.
