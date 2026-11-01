"""Pure-Python Parser for Video_dataset_description.xlsx.

This module parses the dataset metadata spreadsheet using only Python standard
library modules (zipfile, xml.etree.ElementTree). No openpyxl or pandas required.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Union
import xml.etree.ElementTree as ET
import zipfile


@dataclass
class SequenceMetadata:
    """Metadata for a video sequence."""

    sequence_name: str
    sensor: str  # "V" or "IR"
    primary_class: str  # "AIRPLANE", "BIRD", "DRONE", "HELICOPTER"
    number: int
    distance_bin: str  # "CLOSE", "MEDIUM", "DISTANT", "UNKNOWN"
    drone_type: str  # "F450", "Hubsan H107D+", "Phantom 4 Pro", etc.
    source: str  # "author" or "youtube"
    is_multi_drone: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class MetadataCatalog:
    """Catalog of sequence metadata indexed by sequence_name."""

    def __init__(self, items: Optional[Dict[str, SequenceMetadata]] = None) -> None:
        self._items: Dict[str, SequenceMetadata] = items or {}

    def __len__(self) -> int:
        return len(self._items)

    def get(self, sequence_name: str) -> SequenceMetadata:
        """Get metadata by sequence name, or fallback to name inference."""
        if sequence_name in self._items:
            return self._items[sequence_name]
        return self.infer_from_name(sequence_name)

    def all_items(self) -> List[SequenceMetadata]:
        return list(self._items.values())

    @classmethod
    def infer_from_name(cls, name_or_stem: str) -> SequenceMetadata:
        """Infer basic metadata from standard naming convention (e.g. V_DRONE_001)."""
        stem = Path(name_or_stem).stem.replace("_LABELS", "")
        m = re.match(r"^(V|IR)_([A-Z]+)_(\d+)$", stem)
        if m:
            sensor = m.group(1)
            primary_class = m.group(2)
            number = int(m.group(3))
            seq_name = f"{sensor}_{primary_class}_{number:03d}"
            return SequenceMetadata(
                sequence_name=seq_name,
                sensor=sensor,
                primary_class=primary_class,
                number=number,
                distance_bin="UNKNOWN",
                drone_type="UNKNOWN",
                source="author",
                is_multi_drone=False,
            )
        return SequenceMetadata(
            sequence_name=stem,
            sensor="UNKNOWN",
            primary_class="UNKNOWN",
            number=0,
            distance_bin="UNKNOWN",
            drone_type="UNKNOWN",
            source="unknown",
            is_multi_drone=False,
        )

    @classmethod
    def from_xlsx(cls, xlsx_path: Union[str, Path]) -> MetadataCatalog:
        """Parse Video_dataset_description.xlsx from disk."""
        path = Path(xlsx_path)
        if not path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {path}")

        with zipfile.ZipFile(path, "r") as z:
            # 1. Parse shared strings table
            shared_strings: List[str] = []
            if "xl/sharedStrings.xml" in z.namelist():
                tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
                # Namespace uri: http://schemas.openxmlformats.org/spreadsheetml/2006/main
                ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for si in tree.findall("s:si", ns):
                    # Text can be in <t> or nested in rich text <r>/<t>
                    t_elem = si.find("s:t", ns)
                    if t_elem is not None and t_elem.text:
                        shared_strings.append(t_elem.text)
                    else:
                        r_texts = [
                            t.text
                            for t in si.findall(".//s:t", ns)
                            if t.text
                        ]
                        shared_strings.append("".join(r_texts))

            # 2. Parse sheet1
            sheet_data = z.read("xl/worksheets/sheet1.xml")
            tree = ET.fromstring(sheet_data)
            ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

            items: Dict[str, SequenceMetadata] = {}

            # Skip header row (row 1)
            for row in tree.findall(".//s:sheetData/s:row", ns):
                row_idx = int(row.get("r", "0"))
                if row_idx <= 1:
                    continue

                col_values: Dict[str, str] = {}
                for c in row.findall("s:c", ns):
                    cell_ref = c.get("r", "")
                    col_letter = re.sub(r"\d+", "", cell_ref)
                    val_elem = c.find("s:v", ns)
                    if val_elem is None or not val_elem.text:
                        continue
                    val_str = val_elem.text.strip()
                    if c.get("t") == "s" and val_str.isdigit():
                        idx = int(val_str)
                        if 0 <= idx < len(shared_strings):
                            col_values[col_letter] = shared_strings[idx]
                    else:
                        col_values[col_letter] = val_str

                sensor = col_values.get("A", "").strip()
                primary_class = col_values.get("B", "").strip()
                num_str = col_values.get("C", "0").strip()
                try:
                    number = int(float(num_str))
                except ValueError:
                    number = 0

                if not sensor or not primary_class or number == 0:
                    continue

                distance_bin = col_values.get("D", "UNKNOWN").strip().upper()
                drone_info = col_values.get("F", "").strip()

                is_youtube = "youtube" in drone_info.lower()
                source = "youtube" if is_youtube else "author"
                drone_type = "" if is_youtube else drone_info
                is_multi = any(k in drone_info.lower() for k in [",", "and"])

                seq_name = f"{sensor}_{primary_class}_{number:03d}"
                items[seq_name] = SequenceMetadata(
                    sequence_name=seq_name,
                    sensor=sensor,
                    primary_class=primary_class,
                    number=number,
                    distance_bin=distance_bin if distance_bin else "UNKNOWN",
                    drone_type=drone_type if drone_type else "standard",
                    source=source,
                    is_multi_drone=is_multi,
                )

        return cls(items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Video_dataset_description.xlsx in pure Python"
    )
    parser.add_argument("xlsx_path", type=str, help="Path to .xlsx file")
    parser.add_argument(
        "--json", action="store_true", help="Dump all records as JSON"
    )
    args = parser.parse_args()

    catalog = MetadataCatalog.from_xlsx(args.xlsx_path)
    print(f"Loaded {len(catalog)} sequence metadata records")

    if args.json:
        dump = {item.sequence_name: item.to_dict() for item in catalog.all_items()}
        print(json.dumps(dump, indent=2))
    else:
        v_items = [it for it in catalog.all_items() if it.sensor == "V"]
        print(f"Video_V sequences: {len(v_items)}")
        dist_counts = {}
        for it in v_items:
            dist_counts[it.distance_bin] = dist_counts.get(it.distance_bin, 0) + 1
        print("Distance bin distribution in Video_V:")
        for k, v in sorted(dist_counts.items()):
            print(f"  {k:10s}: {v}")


if __name__ == "__main__":
    main()
