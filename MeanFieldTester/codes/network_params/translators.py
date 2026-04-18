from typing import Any, Dict
from dataclasses import dataclass
from pydantic import BaseModel

# ==========================================
# 1. THE UNIT CONVERSION ENGINE
# ==========================================

# Standard SI prefixes used in neuroscience
SI_PREFIXES = {
    'k': 1e3,   # kilo
    '': 1.0,    # base
    'm': 1e-3,  # milli
    'u': 1e-6,  # micro
    'n': 1e-9,  # nano
    'p': 1e-12  # pico
}
BASE_UNITS = ['V', 'A', 'F', 'S', 's', 'Hz']  # Extend as needed


def get_unit_multiplier(source_unit: str, target_unit: str) -> float:
    """Calculates the scaling factor between two scientific units."""
    if source_unit == target_unit or not source_unit or not target_unit:
        return 1.0

    def parse_prefix(unit: str) -> tuple[float, str]:
        """Parses a unit string and returns its scale factor and base unit.

        For example, "mV" -> (1e-3, "V"), "nA" -> (1e-9, "A"), "s" -> (1.0, "s")
        """

        if len(unit) > 1 and unit[0] in SI_PREFIXES and unit[1:] in BASE_UNITS:
            return SI_PREFIXES[unit[0]], unit[1:]
        
        return 1.0, unit  # If no prefix (e.g., 'V', 's'), return base scale 1.0

    source_scale, source_base = parse_prefix(source_unit)
    target_scale, target_base = parse_prefix(target_unit)

    if source_base != target_base:
        raise ValueError(f"Cannot convert between different base units: {source_unit} -> {target_unit}")

    return source_scale / target_scale

# ==========================================
# 2. THE RULE STRUCTURE
# ==========================================

@dataclass
class TranslationRule:
    mft_name: str
    sim_unit: str = ""

# ==========================================
# 3. THE SMART TRANSLATOR
# ==========================================

def translate_params(pydantic_model: BaseModel, mapping_rules: Dict[str, TranslationRule]) -> Dict[str, Any]:
    simulator_dict = {}
    
    for sim_key, rule in mapping_rules.items():
        if not hasattr(pydantic_model, rule.mft_name):
            raise AttributeError(f"Mapping failed: '{rule.mft_name}' not found.")
            
        raw_val = getattr(pydantic_model, rule.mft_name)
        
        #NOTE: unitless parameters are simply copied, their rescalling has to be done by hand (e.g. probability? conversion to percentage?)
        if rule.sim_unit:
            if rule.mft_name in pydantic_model.model_fields:
                field_info = pydantic_model.model_fields[rule.mft_name]
            elif rule.mft_name in pydantic_model.model_computed_fields:
                field_info = pydantic_model.model_computed_fields[rule.mft_name]
            else:
                # This should never happen due to the earlier hasattr check, but we add this for safety.
                raise AttributeError(f"Field '{rule.mft_name}' not found in model fields or computed fields.")

            desc = field_info.description or ""
            parts_open = desc.split('[')

            if len(parts_open) == 1:
                raise ValueError(
                    f"Missing unit in description for '{rule.mft_name}'. "
                    f"Simulator expects '{rule.sim_unit}', but MFT model says: '{desc}'"
                )
                
            if len(parts_open) > 2:
                raise ValueError(
                    f"Too many '[' brackets in description for '{rule.mft_name}'. "
                    f"Only one unit bracket is allowed. Found: '{desc}'"
                )

            parts_close = parts_open[1].split(']')

            if len(parts_close) != 2:
                raise ValueError(
                    f"Malformed closing bracket ']' in description for '{rule.mft_name}'. "
                    f"Found: '{desc}'"
                )
                
            mft_unit = parts_close[0].strip()
            
            multiplier = get_unit_multiplier(source_unit=mft_unit, target_unit=rule.sim_unit)
            raw_val = raw_val * multiplier
                
        simulator_dict[sim_key] = raw_val

    return simulator_dict