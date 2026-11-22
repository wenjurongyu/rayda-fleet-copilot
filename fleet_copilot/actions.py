"""Evidence-gated proposals and a transactional simulated executor."""

import json
import uuid

from .domain import Action, FleetError, Principal, Query
from .handling import PENDING, view_handling
from .store import Store, digest, json_text, now
from .telemetry import DEFAULT_DISK_FREE_PERCENT, DEFAULT_RAM_USED_PERCENT, comparison_for, evidence_keys, read_history, read_latest

MAX_PROPOSALS_PER_RUN = 10


def _policy(query: Query, action: Action):
    if action == "ticket" and query.metric == "compliance":
        return "latest"
    if action == "replacement_review" and query.metric == "battery" and query.days >= 5:
        return "history"
    threshold = query.threshold_percent
    if action == "ticket" and query.days >= 5:
        comparison = comparison_for(query)
        if query.metric == "disk" and comparison in ("lt", "le"):
            if threshold is None or threshold < DEFAULT_DISK_FREE_PERCENT or (
                threshold == DEFAULT_DISK_FREE_PERCENT and comparison == "lt"
            ):
                return "history"
        if query.metric == "ram" and comparison in ("gt", "ge"):
            if threshold is None or threshold >= DEFAULT_RAM_USED_PERCENT:
                return "history"
    raise FleetError("This action needs supported evidence and an actionable policy threshold.")


def _candidates(query, action, latest, history):
    required = _policy(query, action)
    if latest is None or (required == "history" and history is None):
        raise FleetError("This action needs fresh evidence from the current request.")
    current = set(latest["matches"])
    if required == "history":
        current.intersection_update(history["matches"])
        return [(device, None) for device in sorted(current)]
    return [
        (item["device"], check["check_id"])
        for item in latest["details"] if item["device"] in current
        for check in item["data"]["checks"] if check["status"] == "fail"
    ]


def draft(store: Store, principal: Principal, run_id: str, query: Query, action: Action,
          observed_latest, observed_history):
    if principal.role != "admin":
        raise FleetError("An administrator account is required to propose actions.")
    candidates = _candidates(query, action, observed_latest, observed_history)
    if not candidates:
        raise FleetError("The available evidence supports no action for this request.")
    required = _policy(query, action)
    records = []
    with store.connect(write=True) as conn:
        as_of = store.as_of(conn, principal)
        candidates = [
            (device, check_id) for device, check_id in candidates
            if not _awaiting_verification(store, conn, principal, as_of, device, query.metric, check_id)
        ]
        if not candidates:
            raise FleetError("Reported handling is awaiting telemetry verification for the affected issues.")
        if len(candidates) > MAX_PROPOSALS_PER_RUN:
            raise FleetError(f"Narrow this request to at most {MAX_PROPOSALS_PER_RUN} actionable items.")
        device_ids = sorted({device for device, _ in candidates})
        latest = read_latest(store, conn, principal, query, as_of, device_ids)
        history = read_history(store, conn, principal, query, as_of, device_ids) if required == "history" else None
        still_supported = set(_candidates(query, action, latest, history))
        for device, check_id in candidates:
            latest_refs = evidence_keys(latest, device)
            history_refs = evidence_keys(history, device) if history else ()
            if (
                (device, check_id) not in still_supported
                or latest_refs != evidence_keys(observed_latest, device)
                or (history and history_refs != evidence_keys(observed_history, device))
            ):
                raise FleetError("Telemetry changed while preparing the proposal. Rerun the query.")
            item = next(row for row in latest["details"] if row["device"] == device)
            if check_id:
                check = next(check for check in item["data"]["checks"]
                             if check["check_id"] == check_id and check["status"] == "fail")
                finding = {"check_id": check_id, "severity": check["severity"], "status": "fail"}
                summary = f"Investigate failing {check_id} check"
            else:
                period = next(row for row in history["details"] if row["device"] == device)
                finding = {"criterion": latest["criterion"], "observed_days": period["observed_days"],
                           "breach_days": period["breach_days"], "window_days": history["window_days"]}
                summary = "Review battery service or replacement options" if action == "replacement_review" else (
                    f"Investigate sustained {query.metric} pressure"
                )
            payload = {
                "action": action, "device": device, "query": query.model_dump(), "check_id": check_id,
                "summary": summary, "finding": finding, "latest": list(latest_refs),
                "history": list(history_refs), "simulation": True,
            }
            encoded = json_text(payload)
            content_digest = digest(encoded)
            previous = conn.execute(
                """SELECT id, status FROM proposals
                   WHERE company=? AND digest=? AND (run_id=? OR status='pending')
                   ORDER BY rowid LIMIT 1""",
                (principal.company, content_digest, run_id),
            ).fetchone()
            if previous:
                store.event(conn, principal, "proposal_reused", previous["id"],
                            {"run_id": run_id, "version": content_digest})
                records.append({**dict(previous), "payload": payload})
                continue
            proposal_id = uuid.uuid4().hex
            conn.execute(
                """INSERT INTO proposals (company, id, actor, run_id, payload, digest)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (principal.company, proposal_id, principal.actor, run_id, encoded, content_digest),
            )
            store.event(conn, principal, "proposal_created", proposal_id,
                        {"run_id": run_id, "device": device, "version": content_digest, "action": action})
            records.append({"id": proposal_id, "status": "pending", "payload": payload})
    return records


def _awaiting_verification(store, conn, principal, as_of, device, metric, check_id):
    reports = view_handling(store, conn, principal, device, as_of, metric, check_id or "")
    return any(report["status"] == PENDING for report in reports)


def _still_supported(store, conn, principal, payload):
    query = Query.model_validate(payload["query"])
    action = payload["action"]
    as_of = store.as_of(conn, principal)
    device = payload["device"]
    if _awaiting_verification(store, conn, principal, as_of, device, query.metric, payload["check_id"]):
        return False
    latest = read_latest(store, conn, principal, query, as_of, [device])
    history = read_history(store, conn, principal, query, as_of, [device]) if payload["history"] else None
    return (
        (device, payload["check_id"]) in _candidates(query, action, latest, history)
        and tuple(payload["latest"]) == evidence_keys(latest, device)
        and (history is None or tuple(payload["history"]) == evidence_keys(history, device))
    )


def review(store: Store, principal: Principal, proposal_id: str, approved: bool):
    failure = None
    with store.connect(write=True) as conn:
        row = conn.execute(
            "SELECT * FROM proposals WHERE company=? AND id=?", (principal.company, proposal_id),
        ).fetchone()
        if principal.role != "admin" or row is None:
            failure = "Proposal unavailable for this account."
            store.event(conn, principal, "review_denied", proposal_id, {"reason": "unavailable"})
        elif digest(row["payload"]) != row["digest"]:
            failure = "Proposal integrity verification failed."
            store.event(conn, principal, "review_denied", proposal_id, {"reason": "integrity"})
        elif row["status"] != "pending":
            store.event(conn, principal, "review_ignored", proposal_id, {"status": row["status"]})
        elif not approved:
            conn.execute(
                "UPDATE proposals SET status='rejected' WHERE company=? AND id=?", (principal.company, proposal_id)
            )
            store.event(conn, principal, "proposal_rejected", proposal_id, {"version": row["digest"]})
        elif not _still_supported(store, conn, principal, json.loads(row["payload"])):
            conn.execute(
                "UPDATE proposals SET status='expired' WHERE company=? AND id=?", (principal.company, proposal_id)
            )
            store.event(conn, principal, "proposal_expired", proposal_id, {"reason": "evidence_changed"})
        else:
            conn.execute(
                "INSERT INTO work_orders VALUES (?, ?, ?, ?)", (principal.company, proposal_id, principal.actor, now())
            )
            conn.execute(
                "UPDATE proposals SET status='executed' WHERE company=? AND id=?", (principal.company, proposal_id)
            )
            store.event(conn, principal, "proposal_approved", proposal_id, {"version": row["digest"]})
            store.event(conn, principal, "simulation_executed", proposal_id, {"effect_id": proposal_id})
    if failure:
        raise FleetError(failure)
    return store.get_proposal(principal, proposal_id)
