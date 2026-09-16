"""Adapters around the weather simulators released with TripleMixer.

The upstream scripts are dataset-generation programs with module globals and
relative file lookups.  This adapter invokes their original simulation
functions while giving them deterministic seeds and explicit asset paths.
"""

import contextlib
import importlib.util
import os
from pathlib import Path
import sys
import types

import numpy as np


@contextlib.contextmanager
def working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TripleMixerWeatherAdapter:
    FOG_LEVELS = {
        "light": ("integral_lookup_tables_seg_light_0.008beta", 0.005, 0.008),
        "moderate": ("integral_lookup_tables_seg_moderate_0.05beta", 0.02, 0.05),
        "heavy": ("integral_lookup_tables_seg_heavy_0.2beta", 0.06, 0.2),
    }
    RAIN_LEVELS = {"light": 5.0, "moderate": 20.0, "heavy": 50.0}
    SNOW_LEVELS = {"light": 0.5, "moderate": 1.5, "heavy": 2.5}

    def __init__(self, triplemixer_root):
        self.root = Path(triplemixer_root).resolve()
        if not (self.root / "LICENSE").is_file():
            raise FileNotFoundError("TripleMixer root is invalid: %s" % self.root)
        self.fog_dir = self.root / "tools" / "fog_sim"
        self.lisa_dir = self.root / "tools" / "rain_sim"
        self._fog = None
        self._lisa_class = None
        self._lisa = {}

    def _fog_module(self):
        if self._fog is None:
            self._fog = load_file_module(
                "triplemixer_fog_simulation",
                self.fog_dir / "fog_simulation.py")
        return self._fog

    def _get_lisa(self, weather):
        if weather not in self._lisa:
            if self._lisa_class is None:
                # SciPy 1.14 removed the legacy scipy.integrate.trapz name,
                # while TripleMixer's released LISA code still imports it.
                # Provide the exact modern equivalent locally instead of
                # downgrading SciPy or editing the vendored upstream source.
                import scipy.integrate as scipy_integrate
                if not hasattr(scipy_integrate, "trapz"):
                    scipy_integrate.trapz = scipy_integrate.trapezoid

                # TripleMixer imports PyMieScatt unconditionally, although its
                # released inference path reads the precomputed mie_q.npz and
                # never calls PyMieScatt.  Accept a missing optional package
                # only when that cache is present.  The placeholder deliberately
                # fails if upstream unexpectedly attempts online Mie computation.
                mie_cache = self.lisa_dir / "mie_q.npz"
                if not mie_cache.is_file():
                    raise FileNotFoundError(
                        "Missing TripleMixer Mie coefficient cache: %s" %
                        mie_cache)
                mie_stub_installed = False
                if importlib.util.find_spec("PyMieScatt") is None:
                    mie_stub = types.ModuleType("PyMieScatt")

                    def cache_only_mie(*_args, **_kwargs):
                        raise RuntimeError(
                            "TripleMixer attempted to recompute Mie "
                            "coefficients even though mie_q.npz was verified")

                    mie_stub.MieQ_withDiameterRange = cache_only_mie
                    sys.modules["PyMieScatt"] = mie_stub
                    mie_stub_installed = True

                # atmos_models.py loads mie_q.npz relative to cwd.
                try:
                    with working_directory(self.lisa_dir):
                        module = load_file_module(
                            "triplemixer_atmos_models",
                            self.lisa_dir / "atmos_models.py")
                finally:
                    if mie_stub_installed:
                        sys.modules.pop("PyMieScatt", None)
                self._lisa_class = module.LISA
            with working_directory(self.lisa_dir):
                self._lisa[weather] = self._lisa_class(atm_model=weather)
        return self._lisa[weather]

    @staticmethod
    def _validate(points, weather, severity):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("points must have shape [N,4]")
        if not np.isfinite(points).all():
            raise ValueError("points contain NaN or Inf")
        if weather not in ("fog", "rain", "snow"):
            raise ValueError("unsupported weather: %s" % weather)
        if severity not in ("light", "moderate", "heavy"):
            raise ValueError("unsupported severity: %s" % severity)
        return np.ascontiguousarray(points)

    def simulate(self, points, weather, severity, seed):
        points = self._validate(points, weather, severity)
        if weather == "fog":
            return self._simulate_fog(points, severity, seed)
        return self._simulate_lisa(points, weather, severity, seed)

    def _simulate_fog(self, points, severity, seed):
        module = self._fog_module()
        folder, alpha, beta = self.FOG_LEVELS[severity]
        lookup = self.fog_dir / folder / "original"
        if not lookup.is_dir():
            raise FileNotFoundError("Missing TripleMixer fog lookup: %s" % lookup)
        module.INTEGRAL_PATH = lookup
        module.RNG = np.random.default_rng(int(seed))
        parameter = module.ParameterSet(
            alpha=alpha, beta=beta, gamma=0.000001)
        augmented, _, _, info, fog_label = module.simulate_fog(
            parameter, points.copy(), noise=10, gain=False,
            noise_variant="v4", hard=True, soft=True)
        fog_label = np.asarray(fog_label, dtype=np.uint8).reshape(-1)
        reliability = (fog_label == 0).astype(np.float32)
        source_index = np.arange(len(augmented), dtype=np.int32)
        source_index[fog_label == 1] = -1
        return {
            "points": np.asarray(augmented[:, :4], dtype=np.float32),
            "reliability": reliability,
            "noise_type": fog_label,
            "source_index": source_index,
            "lost_count": 0,
            "simulator_info": dict(info or {},
                                   backend="triplemixer_fog_simulation"),
        }

    def _simulate_lisa(self, points, weather, severity, seed):
        np.random.seed(int(seed) & 0xFFFFFFFF)
        lisa = self._get_lisa(weather)
        rate = (self.RAIN_LEVELS if weather == "rain" else
                self.SNOW_LEVELS)[severity]
        # Preserve provenance for non-scattered returns. Upstream overwrites
        # semantic labels with 112 for scatterers, so offset source indices to
        # avoid collision with that sentinel.
        provenance = (np.arange(len(points), dtype=np.int64) + 1000)[:, None]
        augmented, augmented_provenance = lisa.augment_mc(
            points.copy(), provenance, rate)
        status = np.asarray(augmented[:, 4], dtype=np.uint8)
        augmented_provenance = np.asarray(
            augmented_provenance).reshape(-1).astype(np.int64)
        reliable = status == 2
        source_index = np.full(len(status), -1, dtype=np.int32)
        source_index[reliable] = (
            augmented_provenance[reliable] - 1000).astype(np.int32)
        return {
            "points": np.asarray(augmented[:, :4], dtype=np.float32),
            "reliability": reliable.astype(np.float32),
            "noise_type": status,
            "source_index": source_index,
            "lost_count": int(len(points) - len(augmented)),
            "simulator_info": {
                "rate": float(rate),
                "backend": ("triplemixer_lisa_rain" if weather == "rain"
                            else "triplemixer_lisa_snow_branch"),
            },
        }
