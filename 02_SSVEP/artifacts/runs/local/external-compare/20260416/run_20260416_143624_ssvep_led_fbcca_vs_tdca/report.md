# External SSVEP FBCCA vs TDCA

- Dataset: `<repo>\02_SSVEP\artifacts\datasets\external\dataset_ssvep_led_github`
- Subjects: `Subject2, Subject3, Subject4, Subject5`
- Window seconds: `2.0`
- Latency trim: `0.14s`

## Aggregate

| Model | Win (s) | Overall Acc | Mean Fold Acc | Std | Mean Inference (ms) | Trials |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fbcca_fixed_all8 | 2.0 | 0.6708 | 0.6708 | 0.1201 | 2.299 | 480 |
| tdca_like_legacy | 2.0 | 0.3146 | 0.3146 | 0.0736 | 2.437 | 480 |

## Per Subject

| Subject | Model | Win (s) | Mean Acc | Std | Mean Inference (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Subject2 | fbcca_fixed_all8 | 2.0 | 0.6583 | 0.1210 | 2.319 |
| Subject2 | tdca_like_legacy | 2.0 | 0.2833 | 0.0687 | 2.426 |
| Subject3 | fbcca_fixed_all8 | 2.0 | 0.8167 | 0.0500 | 2.284 |
| Subject3 | tdca_like_legacy | 2.0 | 0.3583 | 0.0493 | 2.448 |
| Subject4 | fbcca_fixed_all8 | 2.0 | 0.5583 | 0.0722 | 2.292 |
| Subject4 | tdca_like_legacy | 2.0 | 0.3000 | 0.0471 | 2.434 |
| Subject5 | fbcca_fixed_all8 | 2.0 | 0.6500 | 0.0289 | 2.301 |
| Subject5 | tdca_like_legacy | 2.0 | 0.3167 | 0.0957 | 2.438 |

## Notes

- This comparison uses only stimulation trials (13/17/21 Hz).
- Rest trials are present in the dataset, but not included in the primary metric because FBCCA and TDCA here are evaluated as frequency classifiers, not full async rejectors.
- Split protocol is leave-one-session-out within each subject.
