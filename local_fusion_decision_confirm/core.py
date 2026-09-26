"""Equal-action-count selection for a frozen decision pilot."""
from local_fusion_decision_pilot_v2.core import choose_distinct_rows


def policy_rows(scores, actions, budgets):
    """The confidence comparator gets exactly the learned KEEP action count per frame."""
    maximum = max(budgets)
    learned_keep = choose_distinct_rows(scores['learned'], actions, maximum, True)
    keep_name = f'learned_keep_top{maximum}'
    selected = {keep_name: learned_keep,
                'confidence_matched_keep': choose_distinct_rows(
                    scores['confidence'], actions, len(learned_keep)) if learned_keep else []}
    for budget in budgets:
        selected[f'learned_top{budget}'] = choose_distinct_rows(
            scores['learned'], actions, budget)
        selected[f'confidence_top{budget}'] = choose_distinct_rows(
            scores['confidence'], actions, budget)
    if len(selected[keep_name]) != len(selected['confidence_matched_keep']):
        raise AssertionError('Matched comparator has a different action count')
    return selected
