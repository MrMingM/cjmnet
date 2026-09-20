"""GT-free local detection repair prototype.

The package is isolated from the frozen GSPR/AttFuse and Stage-3 audit code.
"""

from .model import LocalRepairNet, RepairSelector

__all__ = ["LocalRepairNet", "RepairSelector"]
