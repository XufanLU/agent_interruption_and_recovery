# Multi-Agent Debug Flow

This folder contains a second interruption/recovery example for the ICSE
artifact. It is intentionally different from the XAS checkpoint workflow:

- two roles: a developer agent and a reviewer agent
- mutable software artifacts: source files and tests
- a review/feedback loop rather than a single linear pipeline
- a strong correctness oracle: the final unit tests must pass
- a deterministic dry-run mode for cheap regression tests
- an OpenAI Agents SDK mode for real model/token measurements

The workflow models a small debugging task:

```text
bug report
  -> developer diagnoses the failure
  -> developer writes patch v1
  -> developer runs tests
  -> reviewer rejects patch v1 with feedback
  -> interruption is injected
  -> developer revises patch
  -> developer runs tests
  -> reviewer approves
```

## Recovery Comparison

The A/B experiment uses one injected interruption point:

```text
after reviewer feedback, before developer revision
```

Arm A is the restart baseline:

```text
interruption -> discard checkpoint -> restart from the bug report
```

Arm B is checkpoint resume:

```text
interruption -> load checkpoint -> resume from reviewer feedback
```

The checkpoint stores the current stage, diagnosis, patch version, modified
files, test output, reviewer feedback, next agent, and next action. Resume
validates that the workspace still matches the checkpoint before continuing.

## Metrics

Each arm records:

- wall-clock time
- estimated input/output/total tokens
- diagnoses
- file reads
- file writes
- test runs
- review rounds
- patches attempted
- repeated diagnoses
- duplicated patch writes

These metrics make the workflow comparable to the existing XAS A/B harness while
also exercising software-engineering-specific concerns such as file mutation,
review gates, and patch/test loops.

## Deterministic Dry Run

From the repository root:

```bash
python3 -m pytest -q multiagent_debug_flow
```

Run one experiment and print JSON metrics:

```bash
python3 -m multiagent_debug_flow.dummy_workflow --task-id normalize_scores
python3 -m multiagent_debug_flow.dummy_workflow --task-id parse_duration
```

The generated workspaces and checkpoints are written under `.debug_flow_runs/`
by default. A compact A/B summary is appended to `debug_ab_results.jsonl` by
default. Use `--log-path path/to/results.jsonl` to choose another file, or
`--no-log` for throwaway local runs.

## OpenAI Agents SDK Run

The OpenAI path uses real OpenAI Agents SDK agents:

- `Debug Developer`
  - tools: `read_debug_file`, `write_patch_version`, `run_debug_tests`
- `Debug Reviewer`
  - tool: `review_current_patch`

Install the same agent dependencies used by the XAS OpenAI-agent test:

```bash
python3 -m pip install -r ab_test/requirements.txt
```

Set your API key and model:

```bash
export OPENAI_API_KEY="..."
export DEBUG_FLOW_MODEL="gpt-5.4"
```

Run the OpenAI A/B experiment directly:

```bash
python3 -m multiagent_debug_flow.workflow --task-id normalize_scores
```

By default this also appends agent-level process logs to
`test_logs/agent_process_logs.jsonl`. Each agent run records the arm, stage,
agent name, prompt/input, final output, token usage, and an `experiment_id` that
groups events from the same A/B run. Use
`--process-log-path path/to/log.jsonl` to choose another file, or
`--no-process-log` to disable process logging.

Or run the opt-in pytest:

```bash
RUN_REAL_DEBUG_FLOW_AGENT_TEST=1 \
python3 -m pytest -q -s multiagent_debug_flow/test_openai_agents.py
```

The OpenAI run keeps the same recovery structure as the dry run, but token usage
comes from the Agents SDK result metadata rather than estimates.

## Summarize Results

Both the dummy and OpenAI workflows write the same JSONL schema, so they can be
summarized together or in separate files.

```bash
python3 -m multiagent_debug_flow.check_debug_ab_significance
python3 -m multiagent_debug_flow.check_debug_ab_significance --results path/to/results.jsonl
```

The summary reports B - A differences for wall time, tokens, test runs, agent
runs when present, repeated diagnoses, and file writes.

## SWE-bench Lite Subset

`swe_subset/selected_instances.json` stores 12 randomly sampled SWE-bench Lite
metadata entries for a later realistic extension of this workflow. It does not
include full repository checkouts or Docker images.
