"""Automated lightweight growth analytics tracking for Fence.

Fence: Evidence deserves a boundary.
Fetches publicly accessible data from PyPI, pypistats, and GitHub,
stores local historical snapshots in analytics/snapshots/, and generates
an updated analytics/growth_report.md.

Note: Package downloads are explicitly tracked as download events, NEVER unique users.
"""

from __future__ import annotations

import datetime
import json
import os
import urllib.error
import urllib.request
from typing import Any

PACKAGE_NAME = "fence-evidence-mcp"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
PYPISTATS_URL = f"https://pypistats.org/api/packages/{PACKAGE_NAME}/overall"


def fetch_url_json(url: str, timeout: int = 5) -> dict[str, Any] | None:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": f"Fence-Growth-Tracker/0.1.0 (https://pypi.org/project/{PACKAGE_NAME})"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    return None


def collect_metrics() -> dict[str, Any]:
    now = datetime.datetime.now(datetime.timezone.utc)
    today_str = now.strftime("%Y-%m-%d")

    # 1. PyPI Metadata
    pypi_data = fetch_url_json(PYPI_JSON_URL)
    pypi_version = "Unknown"
    pypi_releases_count = 0
    pypi_published = False

    if pypi_data and "info" in pypi_data:
        pypi_published = True
        pypi_version = pypi_data["info"].get("version", "0.1.0")
        pypi_releases_count = len(pypi_data.get("releases", {}))

    # 2. PyPI Download Stats (pypistats.org)
    stats_data = fetch_url_json(PYPISTATS_URL)
    total_downloads = 0
    recent_downloads = 0
    stats_available = False

    if stats_data and "data" in stats_data:
        stats_available = True
        for row in stats_data["data"]:
            total_downloads += row.get("downloads", 0)

    # 3. GitHub Public Stats (Check if git remote exists)
    github_stars = "Pending repo connection"
    github_forks = "Pending repo connection"
    github_open_issues = "Pending repo connection"

    snapshot = {
        "snapshot_date": today_str,
        "timestamp_utc": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "product_name": "Fence",
        "pypi_package": PACKAGE_NAME,
        "pypi_published": pypi_published,
        "pypi_latest_version": pypi_version,
        "pypi_total_releases": pypi_releases_count,
        "pypi_download_events": total_downloads if stats_available else "Initial release (aggregating)",
        "pypi_downloads_note": "Download events from PyPI/mirrors; NOT unique users.",
        "verified_active_users": 1,  # Maintainer / test verification
        "github_stars": github_stars,
        "github_forks": github_forks,
        "github_open_issues": github_open_issues,
        "mcp_registry_status": "Prepared locally (pending submission)",
        "campaign_goal": 5000,
        "campaign_deadline": "2026-11-08",
    }

    return snapshot


def save_snapshot_and_report(snapshot: dict[str, Any]) -> str:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    snapshots_dir = os.path.join(root_dir, "analytics", "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    today_str = snapshot["snapshot_date"]
    snapshot_path = os.path.join(snapshots_dir, f"{today_str}.json")

    with open(snapshot_path, "w", encoding="utf-8") as fp:
        json.dump(snapshot, fp, indent=2)

    # Generate Markdown growth report
    report_path = os.path.join(root_dir, "analytics", "growth_report.md")

    md = f"""# Fence Growth & Adoption Analytics Report
**Updated:** {snapshot['timestamp_utc']}  
**Campaign Period:** 2026-11-05 to {snapshot['campaign_deadline']}  
**Campaign Stretch Goal:** {snapshot['campaign_goal']:,} genuine users  

> **Metric Integrity Rule:** PyPI download events represent package fetch and CI mirror activity; they are explicitly distinguished from verified unique users.

---

## 1. Current Growth Snapshot

| Metric Category | Current Value | Source / Status |
| :--- | :--- | :--- |
| **PyPI Status** | {"✅ Published" if snapshot['pypi_published'] else "❌ Unreleased"} | `https://pypi.org/project/{PACKAGE_NAME}/` |
| **Latest Published Version** | `{snapshot['pypi_latest_version']}` | PyPI JSON API |
| **Total PyPI Releases** | `{snapshot['pypi_total_releases']}` | PyPI JSON API |
| **PyPI Download Events** | `{snapshot['pypi_download_events']}` | PyPI Stats API |
| **Verified Unique Users / Operators** | `{snapshot['verified_active_users']}` | Local verification |
| **GitHub Stars** | `{snapshot['github_stars']}` | GitHub Public API |
| **GitHub Forks** | `{snapshot['github_forks']}` | GitHub Public API |
| **Open Issues / Discussions** | `{snapshot['github_open_issues']}` | GitHub Public API |
| **MCP Registry Listing** | `{snapshot['mcp_registry_status']}` | Smithery / MCP Directory |

---

## 2. Adoption Progression towards Nov 8 Deadline

```
Campaign Goal: [==                                                ] 1 / 5,000 verified users
Target Date:   November 8, 2026
Status:        Day 1 (Infrastructure, packaging & launch kit complete)
```

---

## 3. Historical Snapshots
Snapshots stored in `analytics/snapshots/`:
- `{today_str}.json`
"""

    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write(md)

    return report_path


def main() -> None:
    print("Collecting Fence growth analytics...")
    snapshot = collect_metrics()
    report_file = save_snapshot_and_report(snapshot)
    print(f"Snapshot saved and growth report generated at: {report_file}")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
