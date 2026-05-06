# A/B Checkpoint Evaluation

This folder contains the code used for the checkpointing A/B evaluation.

The folder includes the minimal first-shell agent definition in `first_shell_agent.py`.
It imports the project scientific workflow modules (`function_calling`, `checkpoint`,
and `usage_tracking`) so the result is reproducible against the same implementation
used by the application.

## Folder Structure


- `requirements.txt`
  - Python dependencies for the simulated and live A/B tests.

- `first_shell_agent.py`
  - Minimal agent needed for the live A/B test.
  - Keeps the examiner-facing evaluation independent from the larger application agent prompt.

- `test_simulated_checkpoint_ab.py`
  - Fast deterministic checkpoint-vs-restart test.

- `test_live_agent_ab.py`
  - Opt-in live test that runs the local first-shell agent through the Agents SDK.

- `check_ab_significance.py`
  - Summarizes `live_ab_results.jsonl` and runs paired t-tests on the per-pair `B - A` wall-time and token differences.

- `test_fixture_pairs.py`
  - Validates the fixture manifest and the local material/spectrum lookup helpers.

- `fixture_pairs.py`
  - Loads `fixtures/xas_cif_pairs.json` and resolves fixture file paths.

- `fixtures/`
  - Manifest-backed CIF/XAS fixture pairs.
  - `xas_cif_pairs.json` is the source of truth for pair IDs, material IDs, formulas, source metadata, and fixture filenames.

- `function_calling/`
  - Minimal vendored workflow package used by the tests.
  - Currently contains the first-shell fitting workflow in `fit.py`.

- `checkpoint.py`, `conftest.py`, `data_paths.py`, `storage.py`, `material_database.py`, `spectrum_database.py`, `usage_tracking.py`
  - Local helper modules required by the vendored workflow and test harness.

- `live_ab_results.jsonl`
  - Default JSONL output for live A/B pair summaries.

- `.abtest/`
  - Optional local virtual environment directory.
  - Ignored by git and not part of the source layout.

## Tests

- `test_simulated_checkpoint_ab.py`
  - Fast deterministic test.
  - Uses the real checkpoint workflow (`orchestrate_first_shell_fit_with_checkpoints`) with monkeypatched fitting internals.
  - Does not call the LLM or spend API tokens.
  - Simulates one interrupted baseline run and one interrupted checkpoint-resume run.

- `test_live_agent_ab.py`
  - Opt-in live test.
  - Uses the local first-shell agent copy (`first_shell_agent.create_first_shell_agent`) + `Runner`.
  - Measures real wall-clock time and real model token usage exposed by the Agents SDK.
  - Spends real model/API tokens.

- `check_ab_significance.py`
  - Offline significance check for live A/B results.
  - Treats each fixture pair as one paired observation.
  - Tests whether the mean `B - A` difference differs from zero using a two-sided paired t-test.
  - Also reports the mean, standard deviation, and 95% t confidence interval.

- `first_shell_agent.py`
  - Minimal copied agent factory needed for the live A/B test.
  - Keeps the examiner-facing evaluation independent from the larger application agent prompt.

- `test_fixture_pairs.py`
  - Fast manifest integrity test.
  - Verifies every declared fixture file exists and the local material/spectrum databases are backed by the manifest.

- `conftest.py`
  - Puts this folder first on `sys.path`, then the parent project path, so local A/B modules win over similarly named backend modules.

- `fixture_pairs.py`
  - Loads fixture metadata from `fixtures/xas_cif_pairs.json`.
  - Supports lookup by fixture pair ID, material ID, or formula.

- `fixtures/`
  - Contains the manifest-backed CIF/XAS fixture pairs.
  - `xas_cif_pairs.json` is the manifest used by tests and local lookup helpers.

- `function_calling/`
  - Vendored copy of the scientific workflow code needed by the tests.
  - Includes the first-shell fitting workflow used by the A/B harness.

- `checkpoint.py`, `data_paths.py`, `storage.py`, `material_database.py`, `spectrum_database.py`, `usage_tracking.py`
  - Minimal local helper modules required by the vendored workflow and test harness.

## Environment Requirements

Use the same Python environment as the project. The simulated test needs `pytest`.
The live test also needs the project scientific dependencies and the OpenAI Agents SDK.

Create the virtual environment with Python 3.11 or newer:

```bash
python3.11 -m venv .abtest
source .abtest/bin/activate
```

Install the A/B test requirements from this folder:

```bash
python3 -m pip install -r requirements.txt
```

For the live test, set in .env file:
```bash
OPENAI_API_KEY="your-openai-api-key"
```
Do not commit `.env` or publish real keys.

We ran these experiments with GPT-5.4. To use another model, set `AB_TEST_MODEL`.

## Run The Simulated Test

From this folder:

```bash
python3 -m pytest -q test_simulated_checkpoint_ab.py
```

## Run The Live Agent Test

From this folder:

```bash
RUN_REAL_AGENT_AB_TEST=1 \
python3 -m pytest -q -s test_live_agent_ab.py
```

Use `-s` so pytest prints the `LIVE_AGENT_AB_METRICS` JSON block.
The live test also emits one compact structured log line per data pair:

```bash
LIVE_AGENT_AB_PAIR_RESULT={...}
```

In that record, arm `A` is the baseline restart behavior and arm `B` is the
checkpoint resume behavior. Each arm includes wall-clock time, token usage, and
tool-call counts when the Agents SDK exposes tool-call items.

Those per-pair records are also appended to one JSONL file by default:

```bash
live_ab_results.jsonl
```

To save to a different JSONL file instead, set `LIVE_AB_LOG_PATH`:

```bash
LIVE_AB_LOG_PATH=./live_ab_results.jsonl \
RUN_REAL_AGENT_AB_TEST=1 \
python3 -m pytest -q -s test_live_agent_ab.py
```

The live test defaults to:

- `LIVE_AB_PAIR_ID=cuo_mp-14549_nims-cu-k`

You can run another fixture pair by setting `LIVE_AB_PAIR_ID`, for example:

```bash
LIVE_AB_PAIR_ID=nio_mp-19009_nims-ni-k \
RUN_REAL_AGENT_AB_TEST=1 \
python3 -m pytest -q -s test_live_agent_ab.py
```

You can still override `LIVE_AB_MATERIAL_ID` or `LIVE_AB_XAS_PATH` directly if needed.

To run every fixture pair in `fixtures/xas_cif_pairs.json`, set `LIVE_AB_ALL_PAIRS=1`:

```bash
LIVE_AB_ALL_PAIRS=1 \
RUN_REAL_AGENT_AB_TEST=1 \
python3 -m pytest -q -s test_live_agent_ab.py
```

All-pairs mode ignores `LIVE_AB_MATERIAL_ID` and `LIVE_AB_XAS_PATH` overrides so each run uses the manifest's own material and XAS file.

## Run The Significance Check

After generating `live_ab_results.jsonl`, run:

```bash
python3 check_ab_significance.py
```

The script computes paired differences as `B - A`, where `A` is the
restart-from-zero baseline and `B` is checkpoint resume. For wall-clock time, a
negative mean means checkpoint resume was faster. For token usage, a positive
mean means checkpoint resume used more tokens. The reported paired t-test checks
whether the mean paired difference is significantly different from zero at
`alpha = 0.05`.
