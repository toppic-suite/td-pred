#!/usr/bin/env python3
"""Parse an msalign file and extract per-scan metadata into a TSV file."""

import sys
import csv

FIELDS = [
    "DATABASE_SEQUENCE",
    "PRECURSOR_CHARGE",
    "INSTRUMENT",
    "ACTIVATION",
    "COLLISION_ENERGY",
]


def parse_msalign(input_path: str, output_path: str) -> None:
    scans = []
    current: dict | None = None

    with open(input_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "BEGIN IONS":
                current = {}
            elif line == "END IONS":
                if current is not None:
                    scans.append(current)
                current = None
            elif current is not None and "=" in line:
                key, _, value = line.partition("=")
                if key in FIELDS:
                    if key == "PRECURSOR_CHARGE":
                        parts = key.split(":")
                        key = parts[0]
                    current[key] = value

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for scan in scans:
            writer.writerow({field: scan.get(field, "") for field in FIELDS})

    print(f"Wrote {len(scans)} scans to {output_path}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "example.msalign"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output.tsv"
    parse_msalign(input_file, output_file)
