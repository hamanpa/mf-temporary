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

def get_unit_multiplier(mft_unit: str, sim_unit: str) -> float:
    """Calculates the scaling factor between two scientific units."""
    if mft_unit == sim_unit or not mft_unit or not sim_unit:
        return 1.0

    def parse_prefix(unit: str) -> tuple[float, str]:
        """Parses a unit string and returns its scale factor and base unit.

        For example, "mV" -> (1e-3, "V"), "nA" -> (1e-9, "A"), "s" -> (1.0, "s")
        """

        if len(unit) > 1 and unit[0] in SI_PREFIXES and unit[1:] in ['V', 'A', 'F', 'S', 's', 'Hz']:
            return SI_PREFIXES[unit[0]], unit[1:]
        
        return 1.0, unit  # If no prefix (e.g., 'V', 's'), return base scale 1.0

    mft_scale, mft_base = parse_prefix(mft_unit)
    sim_scale, sim_base = parse_prefix(sim_unit)

    if mft_base != sim_base:
        raise ValueError(f"Cannot convert between different base units: {mft_unit} -> {sim_unit}")

    return mft_scale / sim_scale

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
            
            multiplier = get_unit_multiplier(mft_unit, rule.sim_unit)
            raw_val = raw_val * multiplier
                
        simulator_dict[sim_key] = raw_val

    return simulator_dict