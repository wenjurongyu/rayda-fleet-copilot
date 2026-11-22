"""Record reported handling separately from telemetry and reconcile by collection time."""

import uuid

from .domain import FleetError, Principal, Query
from .store import Store, now, utc_stamp
from .telemetry import measure_snapshot

HANDLED_METRICS = ("disk", "ram", "battery", "compliance")
PENDING = "pending_verification"


def _recent(conn, principal, device, metric=None, check_id=None):
    sql = "SELECT * FROM handling_reports WHERE company=? AND device=?"
    args = [principal.company, device]
    if metric is not None:
        sql += " AND metric=?"
        args.append(metric)
    if check_id is not None:
        sql += " AND check_id=?"
        args.append(check_id)
    return conn.execute(sql + " ORDER BY occurred_at DESC, recorded_at DESC, id DESC", args)


def _view(record, snapshots):
    result = {key: record[key] for key in ("id", "device", "metric", "actor", "occurred_at", "recorded_at", "baseline")}
    result["check_id"] = record["check_id"] or None
    result["status"] = PENDING
    result["verification"] = None
    query = Query(metric=record["metric"], check_id=result["check_id"])
    for row in snapshots:
        if row["collected"] <= record["occurred_at"]:
            break
        measurement, known, abnormal = measure_snapshot(row, query)
        if known:
            result["status"] = "still_abnormal" if abnormal else "verified"
            result["verification"] = {"device": row["device"], "collected": row["collected"], "measurement": measurement}
            break
    return result


def _device_snapshots(store, conn, principal, device, as_of):
    snapshots = store.snapshots(conn, principal, as_of, [device])
    if not snapshots:
        raise FleetError("Device unavailable in this scope.")
    return snapshots


def record_handling(store: Store, principal: Principal, device: str, metric: str, check_id: str | None,
                    occurred_at: str | None = None):
    if principal.role != "admin":
        raise FleetError("An administrator account is required to record handling.")
    if metric not in HANDLED_METRICS or (metric == "compliance") != bool(check_id) or len(check_id or "") > 80:
        raise FleetError("A compliance handling record requires one check ID. Other metrics use the device alone.")
    check = check_id or ""
    recorded = now()
    try:
        occurred = utc_stamp(occurred_at) if occurred_at else recorded
    except (ValueError, TypeError):
        raise FleetError("Use an ISO timestamp with a time zone for --at.") from None
    if occurred > recorded:
        raise FleetError("Handling time cannot be in the future.")
    with store.connect(write=True) as conn:
        snapshots = _device_snapshots(store, conn, principal, device, store.as_of(conn, principal))
        if check and not any(c["check_id"] == check for row in snapshots for c in row["checks"]):
            raise FleetError("That compliance check is unavailable for this device.")
        previous = _recent(conn, principal, device, metric, check).fetchone()
        if occurred_at is None and previous:
            status = _view(previous, snapshots)
            if status["status"] == PENDING:
                return status
        identical = conn.execute(
            """SELECT * FROM handling_reports
               WHERE company=? AND device=? AND metric=? AND check_id=? AND occurred_at=?""",
            (principal.company, device, metric, check, occurred),
        ).fetchone()
        if identical:
            return _view(identical, snapshots)
        ref = uuid.uuid4().hex
        baseline = next((row["collected"] for row in snapshots if row["collected"] <= occurred), None)
        conn.execute(
            """INSERT INTO handling_reports (company, id, actor, device, metric, check_id, occurred_at, recorded_at, baseline)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (principal.company, ref, principal.actor, device, metric, check, occurred, recorded, baseline),
        )
        store.event(conn, principal, "handling_reported", ref, {
            "device": device, "metric": metric, "check_id": check_id, "occurred_at": occurred, "baseline": baseline,
        })
        event = conn.execute(
            "SELECT * FROM handling_reports WHERE company=? AND id=?", (principal.company, ref)
        ).fetchone()
        return _view(event, snapshots)


def view_handling(store, conn, principal, device, as_of, metric=None, check_id=None):
    records = list(_recent(conn, principal, device, metric, check_id))
    if not records:
        return []
    snapshots = _device_snapshots(store, conn, principal, device, as_of)
    latest_by_issue = {}
    for row in records:
        latest_by_issue.setdefault((row["metric"], row["check_id"]), row)
    return [_view(row, snapshots) for row in latest_by_issue.values()]


def list_handling(store: Store, principal: Principal, device: str):
    with store.connect() as conn:
        as_of = store.as_of(conn, principal)
        _device_snapshots(store, conn, principal, device, as_of)
        return view_handling(store, conn, principal, device, as_of)
