"""Tenant-scoped SQLite storage for snapshots, conversations, and reviews."""

import hashlib
import json
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .domain import Conversation, FleetError, Principal

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    company TEXT NOT NULL,
    device TEXT NOT NULL,
    collected TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (company, device, collected)
);
CREATE TABLE IF NOT EXISTS principals (
    company TEXT NOT NULL,
    actor TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin', 'viewer')),
    token_hash TEXT UNIQUE NOT NULL,
    PRIMARY KEY (company, actor)
);
CREATE TABLE IF NOT EXISTS conversations (
    company TEXT NOT NULL,
    actor TEXT NOT NULL,
    session TEXT NOT NULL,
    state TEXT NOT NULL,
    version INTEGER NOT NULL,
    PRIMARY KEY (company, actor, session)
);
CREATE TABLE IF NOT EXISTS proposals (
    company TEXT NOT NULL,
    id TEXT NOT NULL,
    actor TEXT NOT NULL,
    run_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    digest TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending', 'executed', 'rejected', 'expired')),
    PRIMARY KEY (company, id),
    UNIQUE (company, run_id, digest)
);
CREATE TABLE IF NOT EXISTS work_orders (
    company TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (company, proposal_id),
    FOREIGN KEY (company, proposal_id) REFERENCES proposals(company, id)
);
CREATE TABLE IF NOT EXISTS handling_reports (
    company TEXT NOT NULL,
    id TEXT NOT NULL,
    actor TEXT NOT NULL,
    device TEXT NOT NULL,
    metric TEXT NOT NULL CHECK(metric IN ('disk', 'ram', 'battery', 'compliance')),
    check_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    baseline TEXT,
    PRIMARY KEY (company, id),
    UNIQUE (company, device, metric, check_id, occurred_at)
);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY,
    company TEXT NOT NULL,
    actor TEXT NOT NULL,
    happened_at TEXT NOT NULL,
    event TEXT NOT NULL,
    ref TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_company_ref ON audit(company, ref, seq);
"""


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_stamp(value: str) -> str:
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("collected_at needs a time zone")
    return stamp.astimezone(timezone.utc).isoformat(timespec="microseconds")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _array(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def device_kind(hardware: dict):
    role = hardware.get("form_factor")
    if isinstance(role, str):
        form = role.casefold().strip()
        if form in ("laptop", "notebook", "portable", "convertible"):
            return "laptop"
        if form in ("desktop", "all-in-one", "tower", "mini pc"):
            return "desktop"
    labels = " ".join(str(hardware.get(key) or "") for key in ("model_name", "model_identifier")).casefold()
    if re.search(r"\b(macbook[\w-]*|laptop|notebook|thinkpad|latitude|elitebook|probook|xps[ -]?(1[3-7])|surface (pro|laptop|book))\b", labels):
        return "laptop"
    if re.search(r"\b(mac mini|mac studio|mac pro|imac[\w-]*|desktop|optiplex|thinkcentre|thinkstation|precision tower|xps[ -]?(89\d\d))\b", labels):
        return "desktop"
    return "unknown"


def normalize(raw: dict) -> tuple[str, str, str, str]:
    company, device = raw["company_id"], raw["device_id"]
    if any(not isinstance(field, str) or not field.strip() or len(field) > 80 for field in (company, device)):
        raise ValueError("invalid company or device ID")
    os = _mapping(raw.get("os"))
    memory = _mapping(raw.get("memory"))
    battery = _mapping(raw.get("battery"))
    identity = _mapping(raw.get("device_identity"))
    hardware = {key: identity.get(key) for key in ("model_name", "model_identifier", "form_factor")}
    if hardware["form_factor"] is None:
        hardware["form_factor"] = raw.get("device_type") or raw.get("form_factor")
    # Retain only the fields used by telemetry tools.
    selected = {
        "hardware": hardware, "device_kind": device_kind(hardware),
        "os": {key: os.get(key) for key in ("platform", "product_name", "product_version")},
        "memory": {key: memory.get(key) for key in ("used_memory_bytes", "total_memory_bytes")},
        "battery": {key: battery.get(key) for key in ("battery_present", "condition", "cycle_count", "full_charge_capacity")},
        "volumes": [
            {key: volume.get(key) for key in ("mount_point", "size_bytes", "available_bytes")}
            for volume in _array(raw.get("disk_volumes"))
        ],
        "checks": [
            {key: check.get(key) for key in ("check_id", "status", "severity")}
            for check in _array(raw.get("compliance_results"))
        ],
    }
    return company, device, utc_stamp(raw["collected_at"]), json_text(selected)


class Store:
    def __init__(self, path: Path):
        self.path = path
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self, write=False):
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ingest(self, path: Path):
        inserted = 0
        duplicates = 0
        with self.connect(write=True) as conn, path.open(encoding="utf-8-sig") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    company, device, collected, data = normalize(json.loads(line))
                except (KeyError, TypeError, AttributeError, ValueError) as exc:
                    raise FleetError(f"Invalid snapshot on line {line_no}") from exc
                existing = conn.execute(
                    "SELECT data FROM snapshots WHERE company=? AND device=? AND collected=?",
                    (company, device, collected),
                ).fetchone()
                if existing:
                    if existing["data"] != data:
                        previous = json.loads(existing["data"])
                        enriched = json.loads(data)
                        if "hardware" in previous or previous != {
                            key: value for key, value in enriched.items()
                            if key not in {"hardware", "device_kind"}
                        }:
                            raise FleetError(f"Conflicting snapshot on line {line_no}")
                        conn.execute(
                            """UPDATE snapshots SET data=? WHERE company=? AND device=? AND collected=?""",
                            (data, company, device, collected),
                        )
                    duplicates += 1
                    continue
                conn.execute("INSERT INTO snapshots VALUES (?, ?, ?, ?)", (company, device, collected, data))
                inserted += 1
            tenants = conn.execute(
                """SELECT company, COUNT(DISTINCT device) AS devices, COUNT(*) AS snapshots,
                          MAX(collected) AS as_of
                   FROM snapshots GROUP BY company ORDER BY company"""
            )
            return {"inserted": inserted, "duplicates": duplicates, "companies": [dict(row) for row in tenants]}

    def add_principal(self, company: str, actor: str, role: str):
        token = secrets.token_urlsafe(32)
        with self.connect(write=True) as conn:
            if not conn.execute("SELECT 1 FROM snapshots WHERE company=? LIMIT 1", (company,)).fetchone():
                raise FleetError("Import telemetry for that company first.")
            try:
                conn.execute("INSERT INTO principals VALUES (?, ?, ?, ?)", (company, actor, role, digest(token)))
            except sqlite3.IntegrityError as exc:
                raise FleetError("That account already exists or is invalid.") from exc
        return token

    def authenticate(self, token: str | None) -> Principal:
        if not token:
            raise FleetError("Set FLEET_TOKEN to a provisioned account token.")
        with self.connect() as conn:
            row = conn.execute(
                "SELECT company, actor, role FROM principals WHERE token_hash=?", (digest(token),)
            ).fetchone()
        if row is None:
            raise FleetError("Invalid account token.")
        return Principal(**dict(row))

    def conversation(self, principal: Principal, session: str):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT state, version FROM conversations WHERE company=? AND actor=? AND session=?",
                (principal.company, principal.actor, session),
            ).fetchone()
        return (Conversation.model_validate_json(row["state"]), row["version"]) if row else (Conversation(), 0)

    def save_conversation(self, principal: Principal, session: str, state: Conversation, version: int):
        with self.connect(write=True) as conn:
            if version:
                result = conn.execute(
                    """UPDATE conversations SET state=?, version=version+1
                       WHERE company=? AND actor=? AND session=? AND version=?""",
                    (state.model_dump_json(), principal.company, principal.actor, session, version),
                )
            else:
                result = conn.execute(
                    """INSERT OR IGNORE INTO conversations VALUES (?, ?, ?, ?, 1)""",
                    (principal.company, principal.actor, session, state.model_dump_json()),
                )
            return bool(result.rowcount)

    @staticmethod
    def as_of(conn, principal: Principal):
        return conn.execute(
            "SELECT MAX(collected) FROM snapshots WHERE company=?", (principal.company,)
        ).fetchone()[0]

    @staticmethod
    def snapshots(conn, principal: Principal, as_of: str, devices: list[str] | None = None,
                  earliest: str | None = None):
        args = [principal.company, as_of]
        where = "company=? AND collected<=?"
        if devices is not None:
            if not devices:
                return []
            where += " AND device IN (" + ",".join(["?"] * len(devices)) + ")"
            args.extend(devices)
        if earliest is not None:
            where += " AND collected>=?"
            args.append(earliest)
        rows = conn.execute(
            f"SELECT device, collected, data FROM snapshots WHERE {where} ORDER BY device, collected DESC", args
        )
        return [{"device": row["device"], "collected": row["collected"],
                 **json.loads(row["data"])} for row in rows]

    @staticmethod
    def event(conn, principal: Principal, event: str, ref: str, data: dict):
        conn.execute(
            "INSERT INTO audit (company, actor, happened_at, event, ref, data) VALUES (?, ?, ?, ?, ?, ?)",
            (principal.company, principal.actor, now(), event, ref, json_text(data)),
        )

    def log(self, principal: Principal, event: str, ref: str, data: dict):
        with self.connect(write=True) as conn:
            self.event(conn, principal, event, ref, data)

    def audit(self, principal: Principal, ref: str):
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT actor, happened_at, event, ref, data FROM audit WHERE company=? AND ref=? ORDER BY seq",
                (principal.company, ref),
            )
            return [{**dict(row), "data": json.loads(row["data"])} for row in rows]

    def evidence(self, principal: Principal, device: str, collected: str):
        try:
            collected = utc_stamp(collected)
        except (TypeError, ValueError):
            raise FleetError("Evidence unavailable.") from None
        with self.connect() as conn:
            row = conn.execute(
                "SELECT data FROM snapshots WHERE company=? AND device=? AND collected=?",
                (principal.company, device, collected),
            ).fetchone()
        if row is None:
            raise FleetError("Evidence unavailable.")
        return {"device": device, "collected": collected, **json.loads(row["data"])}

    def get_proposal(self, principal: Principal, proposal_id: str):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM proposals WHERE company=? AND id=?", (principal.company, proposal_id)
            ).fetchone()
            if row is None:
                raise FleetError("Proposal unavailable.")
            effect = conn.execute(
                "SELECT approved_by, created_at FROM work_orders WHERE company=? AND proposal_id=?",
                (principal.company, proposal_id),
            ).fetchone()
        return {
            "id": row["id"], "status": row["status"], "run_id": row["run_id"], "created_by": row["actor"],
            "payload": json.loads(row["payload"]), "simulation": dict(effect) if effect else None,
        }
