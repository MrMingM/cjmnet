def arms(smoke=False):
    items = [(f'{s}-{v}-{budget}',s,'ordinal',v,budget)
             for budget in ('fixed','converge') for s in ('s32','s128','all')
             for v in ('additive','joint')]
    return items[:1] if smoke else items
