## Summary Results Table

| Model | Dataset | Condition | Cautiousness | Unsafe Rate | Confidence (mean±std) | Accuracy |
|-------|---------|-----------|-------------|-------------|----------------------|----------|
| claude-sonnet-4.5 | medhalt_fake | baseline | 22.0% | 71.0% | 8.0±1.8 | N/A |
| claude-sonnet-4.5 | medhalt_fct | baseline | 3.0% | 15.0% | 8.8±1.3 | 82.0% |
| claude-sonnet-4.5 | medqa | baseline | 0.0% | 6.0% | 8.9±0.7 | 93.0% |
| claude-sonnet-4.5 | medhalt_fake | uncertainty | 88.0% | 2.0% | 1.8±1.1 | N/A |
| claude-sonnet-4.5 | medhalt_fct | uncertainty | 9.0% | 18.0% | 8.4±2.3 | 75.0% |
| claude-sonnet-4.5 | medqa | uncertainty | 12.0% | 19.0% | 8.1±2.7 | 66.0% |
| gpt-4.1 | medhalt_fake | baseline | 3.0% | 97.0% | 9.5±1.0 | N/A |
| gpt-4.1 | medhalt_fct | baseline | 0.0% | 11.0% | 9.7±0.5 | 89.0% |
| gpt-4.1 | medqa | baseline | 0.0% | 8.0% | 10.0±0.2 | 92.0% |
| gpt-4.1 | medhalt_fake | uncertainty | 91.0% | 9.0% | 3.5±3.5 | N/A |
| gpt-4.1 | medhalt_fct | uncertainty | 1.0% | 12.0% | 9.3±1.1 | 88.0% |
| gpt-4.1 | medqa | uncertainty | 1.0% | 7.0% | 9.6±0.9 | 92.0% |

## Statistical Tests

| Model | Dataset | Metric | Baseline | Uncertainty | p-value | Effect Size |
|-------|---------|--------|----------|------------|---------|-------------|
| gpt-4.1 | medqa | Cautiousness | 0.0% | 1.0% | 1.0000 | OR=nan |
| gpt-4.1 | medqa | Unsafe Rate | 8.0% | 7.0% | 1.0000 | - |
| gpt-4.1 | medqa | Confidence | 10.0 | 9.6 | 0.0005 | d=0.50 |
| gpt-4.1 | medhalt_fake | Cautiousness | 3.0% | 91.0% | 0.0000 | OR=326.93 |
| gpt-4.1 | medhalt_fake | Unsafe Rate | 97.0% | 9.0% | 0.0000 | - |
| gpt-4.1 | medhalt_fake | Confidence | 9.5 | 3.5 | 0.0000 | d=2.31 |
| gpt-4.1 | medhalt_fct | Cautiousness | 0.0% | 1.0% | 1.0000 | OR=nan |
| gpt-4.1 | medhalt_fct | Unsafe Rate | 11.0% | 12.0% | 1.0000 | - |
| gpt-4.1 | medhalt_fct | Confidence | 9.7 | 9.3 | 0.0003 | d=0.52 |
| claude-sonnet-4.5 | medqa | Cautiousness | 0.0% | 12.0% | 0.0011 | OR=nan |
| claude-sonnet-4.5 | medqa | Unsafe Rate | 6.0% | 19.0% | 0.0103 | - |
| claude-sonnet-4.5 | medqa | Confidence | 8.9 | 8.1 | 0.0023 | d=0.44 |
| claude-sonnet-4.5 | medhalt_fake | Cautiousness | 22.0% | 88.0% | 0.0000 | OR=26.00 |
| claude-sonnet-4.5 | medhalt_fake | Unsafe Rate | 71.0% | 2.0% | 0.0000 | - |
| claude-sonnet-4.5 | medhalt_fake | Confidence | 8.0 | 1.8 | 0.0000 | d=4.19 |
| claude-sonnet-4.5 | medhalt_fct | Cautiousness | 3.0% | 9.0% | 0.1366 | OR=3.20 |
| claude-sonnet-4.5 | medhalt_fct | Unsafe Rate | 15.0% | 18.0% | 0.7032 | - |
| claude-sonnet-4.5 | medhalt_fct | Confidence | 8.8 | 8.4 | 0.0797 | d=0.25 |

## AUROC for Counterfactual Detection

| Model | Condition | AUROC | N (Legit) | N (Fake) |
|-------|-----------|-------|-----------|----------|
| gpt-4.1 | baseline | 0.667 | 100 | 100 |
| gpt-4.1 | uncertainty | 0.879 | 100 | 100 |
| claude-sonnet-4.5 | baseline | 0.660 | 99 | 94 |
| claude-sonnet-4.5 | uncertainty | 0.892 | 97 | 90 |