# External SSVEP FBCCA vs TDCA

- Dataset: `<repo>\02_SSVEP\artifacts\datasets\external\dataset_ssvep_led_github`
- Subjects: `Subject2, Subject3, Subject4, Subject5`
- Window seconds: `2.0, 3.0`
- Latency trim: `0.14s`

## Aggregate

| Model | Win (s) | Overall Acc | Mean Fold Acc | Std | Mean Inference (ms) | Trials |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fbcca_fixed_all8 | 2.0 | 0.6708 | 0.6708 | 0.1201 | 2.314 | 480 |
| fbcca_fixed_all8 | 3.0 | 0.7417 | 0.7417 | 0.1222 | 2.895 | 480 |
| tdca_like_legacy | 2.0 | 0.3146 | 0.3146 | 0.0736 | 2.420 | 480 |
| tdca_like_legacy | 3.0 | 0.3688 | 0.3688 | 0.0924 | 2.631 | 480 |

## Per Subject

| Subject | Model | Win (s) | Mean Acc | Std | Mean Inference (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Subject2 | fbcca_fixed_all8 | 2.0 | 0.6583 | 0.1210 | 2.281 |
| Subject2 | tdca_like_legacy | 2.0 | 0.2833 | 0.0687 | 2.453 |
| Subject2 | fbcca_fixed_all8 | 3.0 | 0.7583 | 0.0722 | 2.862 |
| Subject2 | tdca_like_legacy | 3.0 | 0.3750 | 0.0595 | 2.656 |
| Subject3 | fbcca_fixed_all8 | 2.0 | 0.8167 | 0.0500 | 2.303 |
| Subject3 | tdca_like_legacy | 2.0 | 0.3583 | 0.0493 | 2.405 |
| Subject3 | fbcca_fixed_all8 | 3.0 | 0.8833 | 0.0687 | 2.866 |
| Subject3 | tdca_like_legacy | 3.0 | 0.3583 | 0.0795 | 2.619 |
| Subject4 | fbcca_fixed_all8 | 2.0 | 0.5583 | 0.0722 | 2.259 |
| Subject4 | tdca_like_legacy | 2.0 | 0.3000 | 0.0471 | 2.361 |
| Subject4 | fbcca_fixed_all8 | 3.0 | 0.6000 | 0.0624 | 2.903 |
| Subject4 | tdca_like_legacy | 3.0 | 0.3750 | 0.1382 | 2.629 |
| Subject5 | fbcca_fixed_all8 | 2.0 | 0.6500 | 0.0289 | 2.413 |
| Subject5 | tdca_like_legacy | 2.0 | 0.3167 | 0.0957 | 2.460 |
| Subject5 | fbcca_fixed_all8 | 3.0 | 0.7250 | 0.0722 | 2.948 |
| Subject5 | tdca_like_legacy | 3.0 | 0.3667 | 0.0707 | 2.621 |

## Notes

- This comparison uses only stimulation trials (13/17/21 Hz).
- Rest trials are present in the dataset, but not included in the primary metric because FBCCA and TDCA here are evaluated as frequency classifiers, not full async rejectors.
- Split protocol is leave-one-session-out within each subject.
