import json

with open('hl_3.97.json', 'r') as f:
    data = json.load(f)

# Find motifs for 1G1X
motifs_for_1g1x = []
for motif in data:
    alignment = motif.get('alignment', {})
    for key, unit_ids in alignment.items():
        pdb = unit_ids[0].split('|')[0].upper() if unit_ids else ''
        if '1G1X' in pdb:
            motifs_for_1g1x.append(motif)
            print(f"motif_id: {motif['motif_id']}")
            print(f"alignment key: {key}")
            print(f"num unit_ids: {len(unit_ids)}")
            print(f"first few: {unit_ids[:10]}")
            print()

print(f"\nTotal motifs for 1G1X: {len(motifs_for_1g1x)}")