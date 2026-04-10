from hybrid_controller.ssvep import validation_ui as module


def test_build_calibration_trials_counts_targets_and_idle():
    trials = module.build_calibration_trials((8.0, 10.0, 12.0, 15.0), target_repeats=2, idle_repeats=3)

    assert len(trials) == 11
    freq_counts = {}
    idle_count = 0
    trial_ids = set()
    for trial in trials:
        trial_ids.add(trial.trial_id)
        if trial.expected_freq is None:
            idle_count += 1
        else:
            freq_counts[trial.expected_freq] = freq_counts.get(trial.expected_freq, 0) + 1

    assert freq_counts == {8.0: 2, 10.0: 2, 12.0: 2, 15.0: 2}
    assert idle_count == 3
    assert len(trial_ids) == len(trials)


def test_evaluate_profile_quality_reports_expected_rates():
    profile = module.ThresholdProfile(
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=3.0,
        step_sec=0.25,
        enter_score_th=0.03,
        enter_ratio_th=1.2,
        enter_margin_th=0.005,
        exit_score_th=0.025,
        exit_ratio_th=1.1,
        min_enter_windows=2,
        min_exit_windows=2,
    )
    rows = [
        {
            "label": "8Hz",
            "correct": True,
            "top1_score": 0.04,
            "ratio": 1.5,
            "margin": 0.01,
            "normalized_top1": 0.64,
            "score_entropy": 0.35,
        },
        {
            "label": "10Hz",
            "correct": True,
            "top1_score": 0.028,
            "ratio": 1.4,
            "margin": 0.01,
            "normalized_top1": 0.60,
            "score_entropy": 0.40,
        },
        {
            "label": "12Hz",
            "correct": False,
            "top1_score": 0.06,
            "ratio": 1.8,
            "margin": 0.03,
            "normalized_top1": 0.70,
            "score_entropy": 0.30,
        },
        {
            "label": "idle",
            "correct": False,
            "top1_score": 0.01,
            "ratio": 1.05,
            "margin": 0.001,
            "normalized_top1": 0.30,
            "score_entropy": 0.95,
        },
        {
            "label": "idle",
            "correct": False,
            "top1_score": 0.04,
            "ratio": 1.25,
            "margin": 0.006,
            "normalized_top1": 0.42,
            "score_entropy": 0.82,
        },
    ]

    summary = module.evaluate_profile_quality(rows, profile)

    assert summary["non_idle_windows"] == 3.0
    assert summary["control_windows"] == 2.0
    assert summary["idle_windows"] == 2.0
    assert abs(summary["raw_accuracy"] - (2.0 / 3.0)) < 1e-9
    assert abs(summary["control_recall"] - 0.5) < 1e-9
    assert abs(summary["idle_false_positive"] - 0.5) < 1e-9
