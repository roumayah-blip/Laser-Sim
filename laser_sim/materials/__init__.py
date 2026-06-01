from laser_sim.materials.base import Material, load_material, require_pump_cross_sections
from laser_sim.materials.liekki_yb import load_liekki_yb_cross_sections
from laser_sim.materials.yb_glass import YB_GLASS

__all__ = [
    "Material",
    "load_material",
    "require_pump_cross_sections",
    "YB_GLASS",
    "load_liekki_yb_cross_sections",
]
