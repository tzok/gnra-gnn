#!/usr/bin/env python3
"""
Script to parse hl_3.97.json and find the GNRA motif with motif_id HL_37824.7
"""

import json


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
