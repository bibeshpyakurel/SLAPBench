"""Compare supplied frozen-method scores on one identical attempted pair list."""

from itertools import combinations

from .data import InputError
from .evaluation import evaluate, summarize


ARMS = frozenset({"frozen_dinov2", "adapted_dinov2", "verifinger_fusion", "verifinger_sum"})


def validate_arms(names):
    if not 2 <= len(names) <= len(ARMS) or len(set(names)) != len(names) or any(
        name not in ARMS for name in names
    ):
        raise InputError("Comparison requires two or more distinct supported method arms.")


def paired_decisions(rows, scores_a, scores_b, cutoff_a, cutoff_b):
    """Descriptive acceptance differences; failed comparisons remain rejects."""
    result = {}
    for label in ("genuine", "impostor"):
        selected = [pair for pair in rows if pair.label == label]
        cells = {key: 0 for key in ("both_accept", "a_only_accept", "b_only_accept", "both_reject")}
        coverage = {key: 0 for key in ("both_valid", "a_only_valid", "b_only_valid", "neither_valid")}
        joint_a, joint_b = 0, 0
        for pair in selected:
            a, b = scores_a[pair.pair_id], scores_b[pair.pair_id]
            va, vb = a.score is not None, b.score is not None
            aa, ab = va and a.score > cutoff_a, vb and b.score > cutoff_b
            decision = ("both_accept" if aa and ab else "a_only_accept" if aa else
                        "b_only_accept" if ab else "both_reject")
            validity = ("both_valid" if va and vb else "a_only_valid" if va else
                        "b_only_valid" if vb else "neither_valid")
            cells[decision] += 1
            coverage[validity] += 1
            if va and vb:
                joint_a += aa
                joint_b += ab
        n, joint = len(selected), coverage["both_valid"]
        result[label] = {
            "attempts": n, "coverage_counts": coverage,
            "all_attempts_failure_as_reject": {
                **cells,
                "acceptance_rate_difference_b_minus_a":
                    (cells["b_only_accept"] - cells["a_only_accept"]) / n if n else None,
            },
            "supplemental_both_valid_only": {
                "comparisons": joint, "a_accepts": joint_a, "b_accepts": joint_b,
                "acceptance_rate_difference_b_minus_a": (joint_b - joint_a) / joint if joint else None,
            },
        }
    return result


def compare_methods(images, pairs, methods, target_far):
    """Require exact development/test coverage for every arm before comparison.

    The supplied metadata determines identities, labels, hand and partition for
    all arms. Each cutoff comes from that arm's pooled development scores only.
    This does not verify the provenance of the supplied frozen score tables.
    """
    validate_arms(list(methods))
    # evaluate checks metadata, score alignment/status/finite values and both
    # development classes before any paired arithmetic below can consume scores.
    evaluations = {name: evaluate(images, pairs, methods[name], target_far) for name in sorted(methods)}
    image_map = {row.image_id: row for row in images}
    test = [p for p in pairs if image_map[p.image_a].partition == "test"]
    groups = {"pooled": test, **{
        hand: [p for p in test if image_map[p.image_a].hand == hand] for hand in ("left", "right")}}
    scores = {name: {row.pair_id: row for row in rows} for name, rows in methods.items()}
    reports = {}
    for name, report in evaluations.items():
        cutoff = report["threshold"]["cutoff"]
        reports[name] = {
            "threshold": report["threshold"], "development": report["development"],
            "test": {group: summarize([(p.label, scores[name][p.pair_id]) for p in rows], cutoff)
                     for group, rows in groups.items()},
        }
    comparisons = []
    for a, b in combinations(sorted(methods), 2):
        comparisons.append({
            "method_a": a, "method_b": b,
            "test": {group: paired_decisions(rows, scores[a], scores[b],
                                             reports[a]["threshold"]["cutoff"],
                                             reports[b]["threshold"]["cutoff"])
                     for group, rows in groups.items()},
        })
    return {
        "schema_version": 1,
        "scope": "Offline comparison of supplied scores on identical attempted development/test lists; no inference",
        "protocol": next(iter(evaluations.values()))["protocol"],
        "methods": reports, "paired_comparisons": comparisons,
        "limits": [
            "Supplied scores require separate frozen-checkpoint/configuration and execution review; this tool does not certify provenance.",
            "The explicit FAR target must match the reviewed frozen experiment; this offline tool does not authenticate it against original run records.",
            "Every arm covers every attempted development/test comparison, including explicit failures; no intersection filtering for primary rates.",
            "Each arm uses its own pooled development cutoff unchanged for pooled and hand-specific test summaries and both-valid supplements.",
            "Differences are B minus A acceptance rates: higher genuine acceptance raises TAR; higher impostor acceptance raises FAR.",
            "Failed impostors counted as rejects are not evidence of biometric rejection; inspect coverage counts and status summaries.",
            "Both-valid intersections are supplemental reduced populations and may favor methods with selective failures.",
            "ROC/DET and EER are descriptive; test scores do not select thresholds, checkpoints or a winning method.",
            "Reused subjects/images make trials dependent; no independent-trial significance test, confidence interval or population low-FAR bound is claimed.",
            "The proposed subject-resampling uncertainty method still requires statistical review and is not implemented here.",
        ],
    }
