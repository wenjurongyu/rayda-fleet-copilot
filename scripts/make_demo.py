"""Write a labeled synthetic dataset for a quick, offline walkthrough."""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

END = date(2026, 6, 12)
ACME = "acme-001"
OTHER = "globex-002"


def demo_rows():
    for offset in range(6, -1, -1):
        day = (END - timedelta(days=offset)).isoformat()
        devices = [(ACME, "DEMO-A", 6, offset <= 2, True),
                   (ACME, "DEMO-C", 60, False, False),
                   (OTHER, "DEMO-OTHER", 3, True, True)]
        if offset <= 1:
            devices.append((ACME, "DEMO-B", 7, False, False))
        for company, device, free, screen_fail, battery_service in devices:
            item = {
                "company_id": company, "device_id": device, "collected_at": day + "T12:00:00Z",
                "os": {"platform": "darwin", "product_name": "macOS",
                       "product_version": "14.7" if device == "DEMO-A" else "15.1"},
                "device_identity": {"model_name": "Mac mini" if device == "DEMO-C" else "MacBook Air"},
                "memory": {"used_memory_bytes": 91 if device == "DEMO-A" else 55, "total_memory_bytes": 100},
                "disk_volumes": [{"mount_point": "/", "size_bytes": 100, "available_bytes": free}],
                "compliance_results": [
                    {"check_id": "screen_lock", "status": "fail" if screen_fail else "pass", "severity": "medium"},
                    {"check_id": "disk_encryption", "status": "pass", "severity": "high"},
                ],
            }
            if battery_service:
                item["battery"] = {"battery_present": True, "condition": "Service Recommended",
                                   "cycle_count": 580, "full_charge_capacity": 3400}
            yield item


def write_dataset(path: Path, rows=None):
    rows = demo_rows() if rows is None else rows
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            print(json.dumps(row), file=stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    write_dataset(parser.parse_args().path)
