#!/usr/bin/env python3
"""
Script to parse hl_3.97.json and find the GNRA motif with motif_id HL_37824.7
"""

import json
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class UnitID:
    """Represents a Unit ID with all its components"""

    pdb_id: str  # 4 characters, case-insensitive
    model_number: int  # integer, range 1-99
    chain_id: str  # string, case-sensitive
    residue_id: str  # 1-3 characters, case-insensitive
    residue_number: int  # integer, range: -999..9999
    atom_name: Optional[str] = (
        ""  # 0-4 characters, case-insensitive, blank means all atoms
    )
    alternate_id: Optional[str] = ""  # One of ['A', 'B', 'C', '0'], case-insensitive
    insertion_code: Optional[str] = ""  # 1 character, case-insensitive
    symmetry_operation: Optional[str] = "1_555"  # 5-6 characters, case-insensitive


def parse_unit_id(unit_id_str: str) -> UnitID:
    """Parse a Unit ID string into a UnitID object"""
    parts = unit_id_str.split("|")

    if len(parts) < 5:
        raise ValueError(f"Invalid Unit ID format: {unit_id_str}")

    return UnitID(
        pdb_id=parts[0],
        model_number=int(parts[1]),
        chain_id=parts[2],
        residue_id=parts[3],
        residue_number=int(parts[4]),
        atom_name=parts[5] if len(parts) > 5 else "",
        alternate_id=parts[6] if len(parts) > 6 else "",
        insertion_code=parts[7] if len(parts) > 7 else "",
        symmetry_operation=parts[8] if len(parts) > 8 else "1_555",
    )


def process_alignment(alignment: Dict) -> Dict[str, List[UnitID]]:
    """Process the alignment field and convert Unit ID strings to UnitID objects"""
    processed_alignment = {}

    for key, unit_id_list in alignment.items():
        if not isinstance(unit_id_list, list) or len(unit_id_list) != 6:
            print(
                f"Warning: Expected list of 6 elements for key '{key}', got {len(unit_id_list) if isinstance(unit_id_list, list) else 'non-list'}"
            )
            continue

        processed_units = []
        for unit_id_str in unit_id_list:
            try:
                unit_id = parse_unit_id(unit_id_str)
                processed_units.append(unit_id)
            except (ValueError, IndexError) as e:
                print(f"Error parsing Unit ID '{unit_id_str}': {e}")
                continue

        processed_alignment[key] = processed_units

    return processed_alignment


def find_gnra_motif():
    """Parse hl_3.97.json and find the object with motif_id HL_37824.7"""

    try:
        # Load the JSON file
        with open("hl_3.97.json", "r") as f:
            data = json.load(f)

        # Search for the object with the specified motif_id
        for obj in data:
            if obj.get("motif_id") == "HL_37824.7":
                print(f"Found GNRA motif with motif_id: {obj['motif_id']}")
                print(json.dumps(obj, indent=2))

                # Process the alignment field
                if "alignment" in obj:
                    print("\nProcessing alignment field...")
                    processed_alignment = process_alignment(obj["alignment"])

                    print(f"\nProcessed {len(processed_alignment)} alignment entries:")
                    for key, unit_ids in processed_alignment.items():
                        print(f"\n{key}:")
                        for i, unit_id in enumerate(unit_ids):
                            print(f"  {i + 1}. {unit_id}")

                return obj

        print("No object found with motif_id HL_37824.7")
        return None

    except FileNotFoundError:
        print("Error: hl_3.97.json file not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in hl_3.97.json")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    find_gnra_motif()
