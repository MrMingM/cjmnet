"""Retain original inference.main, native GT, postprocess and O/O*/F replay."""
from pathlib import Path
import sys
from bridge import load
from auditlib import atomic_json, sha256

_engine = load('repe_evaluation', 'gspr_repe/evaluation.py')
inherited = _engine.inherited
selection, clean_ap = _engine.selection, _engine.clean_ap


def evaluate(args, checkpoint, destination, clean_only=False, smoke=False):
    # The frozen evaluator isolates the official runtime before importing models.
    # Importing our model here would import OpenCOOD too early and violate that gate.
    if 'models' in sys.modules:
        sys.modules['models'].OBSERVATIONS.clear()
    result = _engine.evaluate(args, checkpoint, destination, clean_only, smoke)
    result.update(stage='GSPR_GGCA', model_variant=args.variant)
    path = Path(destination) / 'repe_inference.json'
    if path.exists():
        result['output_hashes'][path.name] = sha256(path)
    result['mechanism_coverage'] = 'Recorded uncached forwards; see repe_inference.json. No diagnostics invented for cached frames.'
    atomic_json(Path(destination) / 'summary.json', result)
    return result
