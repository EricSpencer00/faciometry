"""Model backends: protocols, device policy, the weight cache, and four models.

This is the only package in Vitruve that depends on torch, and the dependency
is one-directional: `core`, `measure` and `norms` never import from here, which
is what keeps the measurement layer testable in milliseconds without a GPU.

The public surface is deliberately small. Import the protocols to write against
the boundary, and the registry to obtain a backend:

    from vitruve.models.protocols import FaceBox, LandmarkResult, HeadPose
    from vitruve.models.registry import build_detector, build_landmarker
    from vitruve.models.licensing import Tier

Nothing here imports a backend at module load, so `import vitruve.models` costs
nothing and fails on no missing optional dependency. The heavy imports happen
inside `registry.build_*`, after the licence tier has been checked.
"""

from __future__ import annotations

from .licensing import LicenseViolation, Provenance, Tier, require

__all__ = ["LicenseViolation", "Provenance", "Tier", "require"]
