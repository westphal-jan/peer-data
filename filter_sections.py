import glob
import json
import os

# Filter out submission which sections were not parsed correctly
file_paths = glob.glob(f"./data/original/*.json")
to_remove = []
for file_path in file_paths:
    with open(file_path) as f:
        paper_json = json.load(f)
        sections = paper_json["pdf"]["metadata"]["sections"]
        if not sections:
            to_remove.append(file_path)
print(len(to_remove))

for file_path in to_remove:
    os.remove(file_path)
