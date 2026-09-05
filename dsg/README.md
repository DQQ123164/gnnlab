# NTU RGB+D Five-Class Skeleton Splits

This directory contains raw `.skeleton` files for five selected NTU RGB+D
actions. No NumPy conversion is applied.

## Directory layout

```text
data/datasets/
|-- README.md
|-- xsub/
|   |-- train/*.skeleton
|   `-- test/*.skeleton
`-- xview/
    |-- train/*.skeleton
    `-- test/*.skeleton
```

## Selected actions

| Action ID | Description |
|-----------|-------------|
| A033 | check time |
| A044 | headache |
| A045 | chest pain |
| A046 | back pain |
| A047 | neck pain |

## Protocols and split rules

The source files come from
`data/NTU-RGB-D-5class/nturgb+d_skeletons`.

- `xsub` first selects the official X-Sub training subjects.
- `xview` first selects the official X-View training cameras (2 and 3).
- Each action class is independently shuffled and split approximately 80/20.
- The base random seed is `20260905`; X-View uses `20260906`.
- Files remain in the original NTU `.skeleton` text format.
- Train and test files do not overlap within the same protocol.
- The two protocols are independent, so a file may occur in both `xsub` and
  `xview`.

## File counts

| Protocol | Partition | Total | A033 | A044 | A045 | A046 | A047 |
|----------|-----------|------:|-----:|-----:|-----:|-----:|-----:|
| xsub | train | 2685 | 536 | 536 | 537 | 538 | 538 |
| xsub | test | 670 | 134 | 134 | 134 | 134 | 134 |
| xview | train | 2525 | 504 | 504 | 505 | 506 | 506 |
| xview | test | 630 | 126 | 126 | 126 | 126 | 126 |

The small differences in training counts reflect the available source count
for each action. The test partitions are exactly balanced across all five
actions.


## Preprocessing

Run the unified preprocessor from the GNNLab repository root:

```bash
python3 tools/preprocess_datasets.py --datasets dsg --overwrite
```

The command displays progress for each protocol and partition, parses the raw
skeleton files, and writes ST-GCN-compatible arrays to:

```text
processed/dsg/{xsub,xview}/
```

Each data array is `float32` with shape `(N, 3, 300, 25, 2)`. Labels,
sample names, per-sample manifests, and split metadata are generated alongside
the arrays. Use `--no-progress` to suppress progress output.
