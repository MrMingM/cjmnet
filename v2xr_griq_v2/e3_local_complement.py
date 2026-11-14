VARIANTS = ('additive','lidar','radar','independent','cross','joint_local_capacity')


def arms(smoke=False):
    return [(f'E3-{i}','s128','ordinal',v) for i,v in enumerate(VARIANTS) if not smoke or i in (0,4)]
