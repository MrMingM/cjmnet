"""Validate legacy fusion settings without silently changing coordinates."""


def fusion_args_for_probe(fusion):
    if fusion.get('core_method') != 'IntermediateFusionDataset':
        raise ValueError('Harm probe requires IntermediateFusionDataset')
    args = fusion.get('args', {})
    # Classic OpenCOOD YAML uses args: []. Its dataset defaults to proj_first=True.
    if isinstance(args, list) and not args:
        args = {}
    if not isinstance(args, dict):
        raise ValueError(f'Unsupported fusion.args: {args!r}; expected a mapping or legacy empty list')
    result = dict(args)
    proj_first = result.get('proj_first', True)
    if proj_first is not True:
        raise ValueError(f'Harm probe requires proj_first=True; saved value is {proj_first!r}. '
                         'Do not override a false setting merely to bypass this check.')
    result['proj_first'] = True
    return result
