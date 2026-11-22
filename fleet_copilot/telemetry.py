"""Deterministic measurements and citations used by the graph and approvals."""

from collections import defaultdict
from datetime import datetime, time, timedelta
import math
import operator
import re

from .domain import FleetError, Principal, Query
from .store import Store

DEFAULT_DISK_FREE_PERCENT = 10.0
DEFAULT_RAM_USED_PERCENT = 85.0
FRESH_FOR = timedelta(days=2)


def _ref(row):
    return {"device": row["device"], "collected": row["collected"]}


def _ratio(numerator, denominator):
    if (
        type(numerator) not in (int, float) or type(denominator) not in (int, float)
        or denominator <= 0 or not 0 <= numerator <= denominator
    ):
        return None
    return 100 * numerator / denominator


def _version(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{1,5}(\.\d{1,5}){0,3}", value):
        return None
    return tuple(int(part) for part in value.split(".")) + (0,) * (4 - len(value.split(".")))


_COMPARISONS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}
_SYMBOLS = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}


def comparison_for(query):
    return query.comparison or ("lt" if query.metric == "disk" else "ge")


def _matches_ratio(value, limit, query):
    return value is not None and _COMPARISONS[comparison_for(query)](value, limit)


def _os_matches(version, query):
    cutoff = _version(query.os_before)
    exact = _version(query.os_version)
    return version is not None and (cutoff is None or version < cutoff) and (exact is None or version == exact)


def _reading(row, query: Query):
    match query.metric:
        case "disk":
            platform = row["os"]["platform"]
            system_mount = {"win32": "c:" + chr(92), "darwin": "/", "linux": "/"}.get(platform)
            volume = next((v for v in row["volumes"] if isinstance(v["mount_point"], str)
                           and v["mount_point"].lower() == system_mount), None)
            if volume is None:
                return None, False
            free = _ratio(volume["available_bytes"], volume["size_bytes"])
            limit = DEFAULT_DISK_FREE_PERCENT if query.threshold_percent is None else query.threshold_percent
            return {"free_percent": free, "mount": volume["mount_point"]}, _matches_ratio(free, limit, query)
        case "ram":
            memory = row["memory"]
            used = _ratio(memory["used_memory_bytes"], memory["total_memory_bytes"])
            limit = DEFAULT_RAM_USED_PERCENT if query.threshold_percent is None else query.threshold_percent
            return {"used_percent": used}, _matches_ratio(used, limit, query)
        case "battery":
            battery = row["battery"]
            condition = battery["condition"]
            condition = condition.strip() if isinstance(condition, str) else None
            if battery["battery_present"] is False:
                return {"battery_present": False}, False
            if battery["battery_present"] is not True or not condition:
                return None, False
            return {
                "battery_present": True, "condition": condition, "cycles": battery["cycle_count"],
                "full_charge_capacity": battery["full_charge_capacity"],
            }, condition.casefold() in {"service recommended", "replace soon", "replace now", "service battery"}
        case "compliance":
            readings = defaultdict(list)
            for check in row["checks"]:
                check_id = check["check_id"]
                if (
                    isinstance(check_id, str)
                    and (query.check_id is None or query.check_id == check_id)
                    and check not in readings[check_id]
                ):
                    readings[check_id].append(check)
            checks = []
            ambiguous = []
            for check_id, variants in sorted(readings.items()):
                if len(variants) != 1:
                    ambiguous.append(check_id)
                    continue
                check = variants[0]
                if check["status"] in ("pass", "fail") and (
                    query.severity is None or query.severity == check["severity"]
                ):
                    checks.append(check)
            return {"checks": checks, "ambiguous_checks": ambiguous}, any(check["status"] == "fail" for check in checks)
        case "os":
            os = row["os"]
            version = _version(os["product_version"])
            return {"os": os}, _os_matches(version, query)
        case "inventory":
            return {"os": row["os"]}, True
    raise ValueError("Unsupported metric")


def _known(data, metric):
    if data is None:
        return False
    if metric == "disk":
        return data["free_percent"] is not None
    if metric == "ram":
        return data["used_percent"] is not None
    if metric == "compliance":
        return bool(data["checks"])
    if metric == "os":
        return _version(data["os"]["product_version"]) is not None
    return True


def definition(query: Query) -> str:
    if query.metric == "disk":
        limit = DEFAULT_DISK_FREE_PERCENT if query.threshold_percent is None else query.threshold_percent
        return f"system volume free space {_SYMBOLS[comparison_for(query)]} {limit:g}%"
    if query.metric == "ram":
        limit = DEFAULT_RAM_USED_PERCENT if query.threshold_percent is None else query.threshold_percent
        return f"reported RAM utilization {_SYMBOLS[comparison_for(query)]} {limit:g}%"
    if query.metric == "battery":
        return "reported battery service condition"
    if query.metric == "compliance":
        filters = " / ".join(value for value in (query.severity, query.check_id) if value)
        return f"failed compliance check{f' ({filters})' if filters else ''}"
    if query.metric == "os" and query.os_before:
        return f"OS version earlier than {query.os_before}"
    if query.metric == "os" and query.os_version:
        return f"OS version exactly {query.os_version}"
    return query.metric


def measure_snapshot(row, query: Query):
    data, matches = _reading(row, query)
    return data, _known(data, query.metric), matches


def _group(rows):
    by_device = defaultdict(list)
    for row in rows:
        by_device[row["device"]].append(row)
    return by_device


def read_latest(store: Store, conn, principal: Principal, query: Query, as_of: str, devices=None):
    rows = _group(store.snapshots(conn, principal, as_of, devices))
    if devices is not None and any(device not in rows for device in devices):
        raise FleetError("One or more devices are unavailable in this scope.")
    anchor = datetime.fromisoformat(as_of)
    details = []
    unclassified = []
    unknown_version = []
    for device, snapshots in rows.items():
        latest = snapshots[0]
        if query.platform and latest["os"]["platform"] != query.platform:
            continue
        kind = latest.get("device_kind", "unknown")
        if query.device_kind and kind == "unknown":
            unclassified.append(_ref(latest))
        if query.device_kind and kind != query.device_kind:
            continue
        if query.metric != "os" and (query.os_version or query.os_before):
            version = _version(latest["os"]["product_version"])
            if version is None:
                unknown_version.append(_ref(latest))
            if not _os_matches(version, query):
                continue
        data, known, matches = measure_snapshot(latest, query)
        fresh = anchor - datetime.fromisoformat(latest["collected"]) <= FRESH_FOR
        details.append({"device": device, "data": data, "evidence": [_ref(latest)], "fresh": fresh,
                        "device_kind": kind, "known": known, "matches": matches and fresh and known})
    return {
        "kind": "latest", "as_of": as_of, "criterion": definition(query),
        "device_kind": query.device_kind, "unclassified": unclassified, "unknown_version": unknown_version,
        "scope": {"platform": query.platform, "os_version": query.os_version, "os_before": query.os_before},
        "matches": [item["device"] for item in details if item["matches"]],
        "unknown": [item["device"] for item in details if not item["fresh"] or not item["known"]],
        "details": details,
    }


def _failed(day):
    return {check["check_id"] for check in (day["data"] or {}).get("checks", [])
            if check["status"] == "fail"}


def read_history(store: Store, conn, principal: Principal, query: Query, as_of: str, devices=None):
    latest = read_latest(store, conn, principal, query, as_of, devices)
    included = [item["device"] for item in latest["details"]]
    anchor = datetime.fromisoformat(as_of)
    start = datetime.combine(anchor.date() - timedelta(days=query.days - 1), time.min, tzinfo=anchor.tzinfo)
    samples = _group(store.snapshots(conn, principal, as_of, included, start.isoformat(timespec="microseconds")))
    minimum_observed = max(3, math.ceil(0.7 * query.days))
    minimum_breaches = math.ceil(0.7 * query.days)
    details = []
    for item in latest["details"]:
        device = item["device"]
        by_day = {}
        for row in samples[device]:
            day = row["collected"][:10]
            if day not in by_day:
                data, matches = _reading(row, query)
                by_day[day] = {"ref": _ref(row), "data": data, "matches": matches}
        observed = [entry for _, entry in sorted(by_day.items()) if _known(entry["data"], query.metric)]
        breaches = sum(entry["matches"] for entry in observed)
        sufficient = len(observed) >= minimum_observed
        latest_fails = []
        new_failures = []
        recovered = []
        if query.metric == "compliance" and observed:
            latest_fails = sorted(_failed(observed[-1]))
            first_fails = _failed(observed[0])
            first_pass = {check["check_id"] for check in observed[0]["data"]["checks"] if check["status"] == "pass"}
            last_pass = {check["check_id"] for check in observed[-1]["data"]["checks"] if check["status"] == "pass"}
            new_failures = sorted(set(latest_fails) & first_pass)
            recovered = sorted(first_fails & last_pass)
        details.append({
            "device": device, "observed_days": len(observed), "snapshot_days": len(by_day),
            "breach_days": breaches, "sufficient": sufficient,
            "current_match": item["matches"],
            "matches": sufficient and (breaches >= minimum_breaches or bool(new_failures) or bool(recovered)),
            "new_failures": new_failures, "recovered": recovered,
            "current_failed_checks": latest_fails if item["matches"] else [],
            "evidence": [entry["ref"] for entry in by_day.values()],
        })
    return {
        "kind": "history", "as_of": as_of, "start": start.isoformat(timespec="microseconds"),
        "criterion": definition(query), "window_days": query.days, "minimum_observed": minimum_observed,
        "device_kind": query.device_kind, "unclassified": latest["unclassified"],
        "unknown_version": latest["unknown_version"], "scope": latest["scope"],
        "minimum_breaches": minimum_breaches,
        "matches": [item["device"] for item in details if item["matches"]],
        "unknown": [item["device"] for item in details if not item["sufficient"]],
        "details": details,
    }


def evidence_keys(result, device):
    item = next((item for item in result["details"] if item["device"] == device), None)
    if not item:
        return ()
    return tuple(sorted(reference["collected"] for reference in item["evidence"]))
