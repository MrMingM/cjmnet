"""Existing official fixed-target matching and overlap helpers."""
from .input_audit import BASE, SINGLE
from diagnostics import fixed_metrics
from p0_failure import overlap


def hits(data, weather, index):
    return {str(p['object_id']) for p in data[f'{weather}/{index}']['matched_pairs']['0.7']}
