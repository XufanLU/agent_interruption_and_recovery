# SWE-bench Lite 12-Entry Subset

This folder stores a lightweight random metadata subset for the multi-agent debug-flow experiment.
It does not contain full repository checkouts or Docker images.

- Source dataset: `princeton-nlp/SWE-bench_Lite`
- Source URL: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite
- Config/split: `default` / `test`
- Total rows reported by the Dataset Viewer API: `300`
- Random seed: `20270629`
- Sample size: `12`
- Selected offsets: `[25, 55, 104, 120, 123, 146, 181, 191, 201, 257, 282, 299]`

## Files

- `selected_instances.json`: selected issue metadata, base commits, test oracle fields, and patches from SWE-bench Lite.

## Intended Use

Use these entries as a realistic extension of the local multi-agent debugging workflow:

```text
SWE-bench issue -> developer patch -> tests -> reviewer feedback -> interruption -> restart/resume comparison
```

The local synthetic tasks remain the cheap controlled benchmark. This SWE-bench subset is for external-validity experiments and should be prepared with a SWE-bench-compatible environment when actually running the repositories.
