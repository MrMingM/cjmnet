"""Keep all frozen scientific thresholds; explain unavailable validation precisely."""
from scorw_p7.decision import decide as original_decide


def decide(analysis, scenes, selection, config, smoke=False):
    verdict = original_decide(analysis, scenes, selection, config, smoke)
    if not smoke and verdict["decision"] == "INCONCLUSIVE_VALIDATION_UNAVAILABLE":
        verdict["reason"] = selection["validation_unavailable_reason"]
    return verdict
