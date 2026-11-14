from v2xr_griq_v2.multi_quality import LOSS_WEIGHTS


def arms(smoke=False):
    result = [('E1-1','s32','independent','additive'),('E1-2','s32','ordinal','additive'),
              ('E1-0-capacity','s32','strict','strict_capacity')]
    return result[:2] if smoke else result


def tuning():
    return LOSS_WEIGHTS
