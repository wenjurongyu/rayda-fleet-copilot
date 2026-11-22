"""Local review harness and administrator interface."""

import argparse
import json
import os
import re
import shlex
import sqlite3
import sys
from pathlib import Path

from .actions import review
from .agent import Agent
from .domain import FleetError
from .handling import HANDLED_METRICS, PENDING, list_handling, record_handling
from .model_config import load_model_settings
from .planner import OfflinePlanner, ModelPlanner
from .store import Store


def _json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _citations(item):
    return ", ".join(f"{ref['device']}@{ref['collected']}" for ref in item["evidence"])


def _measurement(item):
    data = item["data"] or {}
    if "free_percent" in data:
        return f"{data['free_percent']:.2f}% free on {data['mount']}"
    if "used_percent" in data:
        return f"{data['used_percent']:.2f}% reported RAM used"
    if "checks" in data:
        facts = [f"{row['check_id']}={row['status']} ({row['severity']})" for row in data["checks"]]
        if data["ambiguous_checks"]:
            facts.append("conflicting readings for " + ", ".join(data["ambiguous_checks"]))
        return ", ".join(facts)
    if "condition" in data:
        return f"{data['condition']}; {data['cycles']} cycles"
    if "battery_present" in data:
        return "No battery reported"
    if "os" in data:
        os_data = data["os"]
        return f"{os_data['product_name'] or os_data['platform']} {os_data['product_version'] or ''}".strip()
    return "Unavailable"


def _ordered_details(result: dict):
    match_order = {device: index for index, device in enumerate(result["matches"])}
    return sorted(result["details"], key=lambda item: match_order.get(item["device"], len(match_order)))


def _render(report: dict, show_trace: bool):
    print(f"{report['company']} | telemetry through {report['as_of']} | run {report['run_id']}")
    if report["reference_group"]:
        print("Follow-up device order")
        for index, device in enumerate(report["reference_group"], 1):
            print(f"  {index}. {device}")
    latest = report["latest"]
    if latest:
        print(f"Latest: {len(latest['matches'])} matching / {len(latest['details'])} examined / "
              f"{len(latest['unknown'])} missing or stale | {latest['criterion']}")
        for item in _ordered_details(latest):
            if not item["fresh"] or not item["known"]:
                message = "stale snapshot" if not item["fresh"] else "measurement unavailable"
                if item["fresh"] and (item["data"] or {}).get("ambiguous_checks"):
                    message = _measurement(item)
            else:
                message = _measurement(item)
            if any(
                handling["device"] == item["device"] and handling["status"] == PENDING
                and item["evidence"][0]["collected"] <= handling["occurred_at"]
                for handling in report["handling_reports"]
            ):
                message += " (snapshot collected before reported handling)"
            print(f"  {item['device']}: {message} [{_citations(item)}]")
    history = report["history"]
    if history:
        print(f"History: {len(history['matches'])} matching / {len(history['details'])} examined / "
              f"{len(history['unknown'])} insufficient | {history['start'][:10]} to {history['as_of'][:10]} UTC")
        print(f"  {history['criterion']}; minimum {history['minimum_observed']} observed days; "
              f"persistence threshold {history['minimum_breaches']} days")
        for item in _ordered_details(history):
            coverage = f"{item['observed_days']}/{history['window_days']} observed days"
            finding = f"{item['breach_days']} matching days" if item["sufficient"] else "insufficient coverage"
            changes = ""
            if item["new_failures"] or item["recovered"]:
                changes = f"; new failures={item['new_failures']}, recovered={item['recovered']}"
            print(f"  {item['device']}: {finding}, {coverage}{changes} [{_citations(item) or 'no snapshots'}]")
    scoped = history or latest
    if scoped:
        scope = scoped["scope"]
        if scope["os_version"] or scope["os_before"]:
            version = scope["os_version"] or f"earlier than {scope['os_before']}"
            print(f"  Scope: {scope['platform']} {version}")
        if scoped["unknown_version"]:
            ids = ", ".join(ref["device"] for ref in scoped["unknown_version"])
            print(f"  OS version unavailable: {ids}")
    if scoped and scoped["device_kind"] and scoped["unclassified"]:
        ids = ", ".join(ref["device"] for ref in scoped["unclassified"])
        print(f"  Device type unavailable for {len(scoped['unclassified'])} device(s): {ids}")
    for proposal in report["proposals"]:
        payload = proposal["payload"]
        print(f"Pending proposal {proposal['id']}: {payload['device']} | {payload['summary']} | simulation")
        print(f"  Review: fleet proposal {proposal['id']}; approve: fleet approve {proposal['id']}")
    for handling in report["handling_reports"]:
        issue = handling["check_id"] or handling["metric"]
        evidence = handling["verification"]
        suffix = f" | verification {evidence['collected']}" if evidence else (
            f" | pre-handling snapshot {handling['baseline']}" if handling["baseline"] else ""
        )
        print(f"Handling {handling['device']} ({issue}): {handling['status']} "
              f"| reported by {handling['actor']} | handled at {handling['occurred_at']}{suffix}")
    if report["notice"]:
        print(report["notice"])
    if report["handling_confirmation"]:
        target = report["handling_confirmation"]
        print("Handling target:", json.dumps(target, ensure_ascii=False))
        fields = (target["device"], target["check_id"] or "none")
        if all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value) for value in fields):
            check = f" --check-id {target['check_id']}" if target["check_id"] else ""
            print(f"Confirm: fleet handled {target['device']} --metric {target['metric']}{check}")
            print("For earlier work, add --at with the actual time and time zone.")
    if show_trace:
        print("\nExecution trace")
        for event in report["trace"]:
            print(json.dumps(event, ensure_ascii=False))


def _parser():
    parser = argparse.ArgumentParser(description="Telemetry-only Fleet Copilot")
    parser.add_argument("--db", default=os.getenv("FLEET_DB", "fleet.sqlite"), help="SQLite file")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("import", help="Import official or synthetic NDJSON (local operator)")
    ingest.add_argument("path", type=Path)
    user = commands.add_parser("user-add", help="Provision an account and print its token (local operator)")
    user.add_argument("--tenant", required=True)
    user.add_argument("--user", required=True)
    user.add_argument("--role", choices=["admin", "viewer"], default="admin")
    for command in ("ask", "chat"):
        ask = commands.add_parser(command, help="Query as the authenticated account")
        if command == "ask":
            ask.add_argument("question")
        ask.add_argument("--session", default="default")
        ask.add_argument("--backend", choices=["offline", "model"], default="offline")
        ask.add_argument("--model-config", type=Path)
        ask.add_argument("--model-profile")
        ask.add_argument("--trace", action="store_true")
        ask.add_argument("--json", action="store_true", dest="as_json")
    for command in ("proposal", "approve", "reject"):
        review_command = commands.add_parser(command)
        review_command.add_argument("proposal_id")
    handled = commands.add_parser("handled", help="Confirm an administrator's handling report")
    handled.add_argument("device")
    handled.add_argument("--metric", choices=HANDLED_METRICS, required=True)
    handled.add_argument("--check-id")
    handled.add_argument("--at", help="ISO time with a zone, defaults to now")
    handling = commands.add_parser("handling", help="View the latest handling record for each issue on a device")
    handling.add_argument("device")
    audit = commands.add_parser("audit")
    audit.add_argument("ref", help="Run or proposal ID")
    evidence = commands.add_parser("evidence")
    evidence.add_argument("device")
    evidence.add_argument("collected", help="UTC timestamp from a citation")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        store = Store(Path(args.db))
        command = args.command
        if command == "import":
            _json(store.ingest(args.path))
            return 0
        if command == "user-add":
            print(store.add_principal(args.tenant, args.user, args.role))
            return 0
        principal = store.authenticate(os.getenv("FLEET_TOKEN"))
        if command in ("ask", "chat"):
            planner = (
                ModelPlanner(load_model_settings(args.model_config, args.model_profile))
                if args.backend == "model" else OfflinePlanner()
            )
            agent = Agent(store, principal, planner)
            if command == "ask":
                report = agent.ask(args.question, args.session)
                if args.as_json:
                    _json(report)
                else:
                    _render(report, args.trace)
                return int(report["status"] == "error")
            while True:
                try:
                    question = input("fleet> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if question.casefold() in ("/quit", "/exit"):
                    break
                if not question:
                    continue
                if question.startswith("/handled "):
                    try:
                        main(["--db", args.db, "handled", *shlex.split(question[len("/handled "):])])
                    except (ValueError, SystemExit):
                        print("Usage: /handled DEVICE_ID --metric METRIC [--check-id ID] [--at UTC]")
                    continue
                report = agent.ask(question, args.session)
                if args.as_json:
                    _json(report)
                else:
                    _render(report, args.trace)
            return 0
        if command == "proposal":
            _json(store.get_proposal(principal, args.proposal_id))
        elif command in ("approve", "reject"):
            _json(review(store, principal, args.proposal_id, command == "approve"))
        elif command == "handled":
            _json(record_handling(store, principal, args.device, args.metric, args.check_id, args.at))
        elif command == "handling":
            _json(list_handling(store, principal, args.device))
        elif command == "audit":
            _json(store.audit(principal, args.ref))
        elif command == "evidence":
            _json(store.evidence(principal, args.device, args.collected))
        return 0
    except (FleetError, OSError, sqlite3.Error) as exc:
        print(f"fleet: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
