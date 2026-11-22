"""Explicit, observable LangGraph plan → decide → tool loop."""

import re
import sqlite3
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from .actions import draft
from .domain import APPROVAL_HINT, Conversation, FleetError, Plan, Principal, Tool
from .handling import HANDLED_METRICS, view_handling
from .planner import OfflinePlanner, Planner, requested_metrics
from .request_rules import (
    UnsupportedScope, action_block_reason, company_references, explicit_threshold_default,
    has_calendar_anchor, incompatible_percentage_unit,
    percentage_terms, references_last_result, requested_devices, requested_os,
)
from .store import Store
from .telemetry import comparison_for, read_history, read_latest

MAX_TOOLS = 4
HANDLING_STATEMENT = re.compile(
    r"^\s*(?:i|we)(?:'ve|’ve|\s+have)?\s+(?:already\s+)?"
    r"(?:handled|fixed|resolved|dealt with|repaired|addressed|taken care of)\b"
    r"|^\s*(?:我|我们)(?:已|已经).{0,30}(?:处理|修复|解决)",
    re.IGNORECASE,
)
HELP = ("I can query your fleet's current telemetry, show trends over time, "
        "and draft evidenced IT proposals for review.")
CLARIFICATIONS = {
    "metric": "Which metric should I check: disk, RAM, battery, compliance, or OS?",
    "device": "Which device should I check? Give its exact ID or select a device from the displayed list.",
    "time_window": "Choose a history window from 2 through 30 UTC days.",
    "request": "Please clarify the fleet metric, filter, or operation you want.",
}


class Turn(TypedDict, total=False):
    run_id: str
    text: str
    memory: Conversation
    as_of: str
    visible_ids: list[str]
    devices: list[str] | None
    plan: Plan
    next_tool: Tool
    latest: dict
    history: dict
    proposals: list[dict]
    handling_confirmation: dict
    attempted_draft: bool
    trace: list[dict]
    tools_used: int
    status: str
    notice: str


def _trace(state: Turn, event: str, **details):
    return [*state["trace"], {"event": event, **details}]


def _resolve(plan: Plan, memory: Conversation, visible_ids):
    id_map = {device.casefold(): device for device in visible_ids}
    group = [device for device in memory.group if device in visible_ids]
    match plan.reference:
        case "fleet":
            return None
        case "ids":
            resolved = [id_map.get(device.casefold()) for device in plan.device_ids]
            if any(device is None for device in resolved):
                raise FleetError("One or more devices are unavailable in this scope.")
            return list(dict.fromkeys(resolved))
        case "focus":
            if memory.focus in visible_ids:
                return [memory.focus]
            if len(group) == 1:
                return group
        case "group":
            if group:
                return group
        case "ordinal":
            if plan.ordinal <= len(group):
                return [group[plan.ordinal - 1]]
        case "last":
            if group:
                return [group[-1]]
        case "other":
            if memory.focus in group:
                others = [device for device in group if device != memory.focus]
                if len(others) == 1:
                    return others
                if others:
                    raise FleetError("Which other device: " + ", ".join(others) + "?")
    raise FleetError(CLARIFICATIONS["device"])


def _choices(state: Turn) -> list[Tool]:
    plan = state["plan"]
    mode = plan.query.mode
    latest = state.get("latest")
    history = state.get("history")
    if latest is None and history is None:
        if mode == "history" and plan.action is None:
            return ["history"] if state["devices"] is not None else ["history", "latest"]
        return ["latest"]
    if mode == "latest_then_history" and latest is not None and not latest["matches"]:
        return ["finish"]
    needs_history = mode != "latest" or (
        plan.action is not None and plan.query.metric in ("disk", "ram", "battery")
    )
    if needs_history and history is None:
        return ["history"]
    if plan.action and latest is None:
        return ["latest"]
    if plan.action and not state.get("attempted_draft") and latest["matches"]:
        return ["propose"]
    return ["finish"]


def _summary(result: dict | None):
    if result is None:
        return None
    return {"matches": result["matches"], "unknown": result["unknown"],
            "devices_examined": len(result["details"]), "unclassified": len(result["unclassified"])}


class Agent:
    def __init__(self, store: Store, principal: Principal, planner: Planner):
        self.store = store
        self.principal = principal
        self.planner = planner
        graph = StateGraph(Turn)
        graph.add_node("plan", self._plan)
        graph.add_node("decide", self._decide)
        graph.add_node("tool", self._tool)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", self._route, {"continue": "decide", "finish": "finish"})
        graph.add_conditional_edges(
            "decide", lambda state: "finish" if state["next_tool"] == "finish" else "tool",
            {"finish": "finish", "tool": "tool"},
        )
        graph.add_conditional_edges("tool", self._route, {"continue": "decide", "finish": "finish"})
        graph.add_edge("finish", END)
        self.graph = graph.compile()

    @staticmethod
    def _route(state):
        return "finish" if state.get("status") in {"declined", "clarify", "error", "help", "confirm_handling"} else "continue"

    def _plan(self, state: Turn):
        company = self.principal.company.casefold()
        company_aliases = {company, company.rsplit("-", 1)[0]}
        if any(reference not in company_aliases for reference in company_references(state["text"])):
            return {"status": "declined", "notice": "This fleet is unavailable in the current account.",
                    "trace": _trace(state, "scope_denied")}
        if HANDLING_STATEMENT.search(state["text"]):
            return self._handling_confirmation(state)
        if has_calendar_anchor(state["text"]):
            return {"status": "clarify", "notice": "Use the current snapshot or a rolling history window in days.",
                    "trace": _trace(state, "unsupported_calendar_anchor")}
        try:
            planned = self.planner.plan(state["text"], state["memory"], state["visible_ids"])
            plan = Plan.model_validate(planned.model_dump())
        except FleetError as exc:
            return {"status": "error", "notice": str(exc), "trace": _trace(state, "plan_error")}
        except ValidationError:
            return {"status": "error", "notice": "The planner returned an inconsistent plan.",
                    "trace": _trace(state, "plan_error", reason="invalid_plan")}
        if plan.intent == "fleet":
            try:
                os_scope = requested_os(state["text"])
            except UnsupportedScope as exc:
                return {"status": "clarify", "notice": str(exc), "trace": _trace(state, "unsupported_scope")}
            if os_scope:
                query = plan.query.model_copy(update={
                    "platform": os_scope.platform, "os_version": os_scope.exact, "os_before": os_scope.before,
                })
                plan = plan.model_copy(update={"query": query})
            explicit = requested_devices(state["text"], state["visible_ids"])
            if explicit:
                if len(explicit) > 40:
                    return {"status": "clarify", "notice": "Select at most 40 devices.",
                            "trace": _trace(state, "unsupported_scope")}
                plan = plan.model_copy(update={"reference": "ids", "device_ids": explicit, "ordinal": None})
            elif references_last_result(state["text"]):
                plan = plan.model_copy(update={"reference": "last", "device_ids": [], "ordinal": None})
            elif plan.action and plan.reference == "ids":
                return {"status": "clarify", "notice": CLARIFICATIONS["device"],
                        "trace": _trace(state, "unconfirmed_target")}
        identifiers = [self.principal.company, self.principal.company.rsplit("-", 1)[0], *plan.device_ids]
        if plan.query and plan.query.check_id:
            identifiers.append(plan.query.check_id)
        action_reason = (
            action_block_reason(state["text"], plan.action, plan.query.metric, identifiers) if plan.action else None
        )
        if action_reason:
            plan = plan.model_copy(update={"action": None})
        trace = _trace(state, "plan", plan=plan.model_dump(exclude_none=True))
        if action_reason:
            trace.append({"event": "action_suppressed", "reason": action_reason})
        intent = plan.intent
        if intent == "help":
            return {"status": "help", "notice": HELP, "trace": trace}
        if intent == "decline":
            return {"status": "declined", "notice": HELP + " " + APPROVAL_HINT, "trace": trace}
        if intent == "clarify":
            return {"status": "clarify", "notice": CLARIFICATIONS[plan.clarification], "trace": trace}
        terms = percentage_terms(state["text"])
        term = terms[0] if len(terms) == 1 else None
        if incompatible_percentage_unit(state["text"], plan.query.metric):
            notice = "Specify percent free for disk or percent used for RAM."
            return {"status": "clarify", "notice": notice,
                    "trace": [*trace, {"event": "unsupported_measurement_basis"}]}
        if plan.query.metric in {"disk", "ram"} and (len(terms) > 1 or (term and (
            plan.query.threshold_percent != term.value
            or term.conflicting
            or (term.comparison is not None and comparison_for(plan.query) != term.comparison)
            or (term.comparison is None and (plan.action or not explicit_threshold_default(state["text"])))
        ))):
            notice = "Confirm the comparison and percentage for free disk space or used RAM."
            return {"status": "clarify", "notice": notice,
                    "trace": [*trace, {"event": "condition_needs_confirmation"}]}
        if plan.company and plan.company.casefold() not in company_aliases:
            return {"status": "declined", "notice": "This fleet is unavailable in the current account.",
                    "trace": [*trace, {"event": "scope_denied"}]}
        try:
            devices = _resolve(plan, state["memory"], state["visible_ids"])
        except FleetError as exc:
            status = "declined" if plan.reference == "ids" else "clarify"
            return {"status": status, "notice": str(exc), "trace": trace}
        result = {"plan": plan, "devices": devices, "trace": trace}
        if action_reason == "unsupported_action_condition":
            result["notice"] = "The action condition could not be verified. Ask for a supported ticket or review with an explicit filter."
        if action_reason == "no_explicit_action_request":
            result["notice"] = "To prepare a proposal, request a ticket or replacement review explicitly."
        return result

    def _handling_confirmation(self, state: Turn):
        trace = _trace(state, "handling_report_requested")
        if self.principal.role != "admin":
            return {"status": "declined", "notice": "An administrator must confirm a handling report.", "trace": trace}
        memory = state["memory"]
        hint = OfflinePlanner().plan(state["text"], memory, state["visible_ids"])
        explicit = requested_devices(state["text"], state["visible_ids"])
        if explicit:
            hint = hint.model_copy(update={"reference": "ids", "device_ids": explicit, "ordinal": None})
        elif hint.reference == "fleet":
            return {"status": "clarify", "notice": CLARIFICATIONS["device"], "trace": trace}
        try:
            devices = _resolve(hint, memory, state["visible_ids"])
        except FleetError as exc:
            return {"status": "clarify", "notice": str(exc), "trace": trace}
        if not devices or len(devices) != 1:
            return {"status": "clarify", "notice": CLARIFICATIONS["device"], "trace": trace}
        if len(set(requested_metrics(state["text"])).intersection(HANDLED_METRICS)) > 1:
            return {"status": "clarify", "notice": "Confirm one handled issue at a time.", "trace": trace}
        query = hint.query or memory.query
        if query is None or query.metric not in HANDLED_METRICS or (
            query.metric == "compliance" and not query.check_id
        ):
            return {"status": "clarify",
                    "notice": "Specify disk, RAM, battery, or the exact compliance check that was handled.", "trace": trace}
        confirmation = {"device": devices[0], "metric": query.metric, "check_id": query.check_id}
        return {"status": "confirm_handling", "handling_confirmation": confirmation, "trace": trace,
                "notice": "Confirm the handling report explicitly. Device telemetry is still evaluated separately."}

    def _decide(self, state: Turn):
        offered = _choices(state)
        latest = _summary(state.get("latest"))
        history = _summary(state.get("history"))
        observations = {"latest": latest, "history": history, "draft_attempted": state.get("attempted_draft", False)}
        if state["tools_used"] >= MAX_TOOLS and offered != ["finish"]:
            return {"next_tool": "finish", "status": "error", "notice": "Tool budget reached.",
                    "trace": _trace(state, "decision", tool="finish", reason="budget_reached", **observations)}
        try:
            # A forced transition expresses a workflow invariant; optional transitions go to the planner.
            tool = offered[0] if len(offered) == 1 else self.planner.next(state["plan"], offered, observations).tool
            if tool == "finish" and offered == ["history", "latest"]:
                return {"next_tool": "history", "trace": _trace(
                    state, "decision", tool="history", basis="history_required_by_plan", offered=offered, **observations
                )}
            if tool not in offered:
                raise FleetError("The planner selected a step outside the allowed workflow.")
        except FleetError as exc:
            return {"next_tool": "finish", "status": "error", "notice": str(exc),
                    "trace": _trace(state, "decision_error", offered=offered, **observations)}
        if tool == "finish" and latest is not None and not latest["matches"] and state["plan"].query.mode == "latest_then_history":
            basis = "no_current_candidates"
        elif tool == "history" and state["plan"].query.mode == "latest_then_history":
            basis = "inspect_history_for_current_candidates"
        else:
            basis = "workflow_constraint" if len(offered) == 1 else "planner_choice"
        return {"next_tool": tool, "trace": _trace(state, "decision", tool=tool, offered=offered,
                                                   basis=basis, **observations)}

    def _tool(self, state: Turn):
        tool = state["next_tool"]
        plan = state["plan"]
        devices = state["devices"]
        if tool == "history" and plan.query.mode == "latest_then_history":
            devices = state["latest"]["matches"]
        params = {"as_of": state["as_of"], "devices": devices, "query": plan.query.model_dump(exclude_none=True)}
        trace = _trace(state, "tool_call", tool=tool, arguments=params)
        update = {"tools_used": state["tools_used"] + 1}
        try:
            if tool == "propose":
                proposals = draft(self.store, self.principal, state["run_id"], plan.query, plan.action,
                                  state.get("latest"), state.get("history"))
                result = {"proposal_ids": [proposal["id"] for proposal in proposals]}
                update.update(proposals=proposals, attempted_draft=True)
            else:
                read = read_latest if tool == "latest" else read_history
                with self.store.connect() as conn:
                    data = read(self.store, conn, self.principal, plan.query, state["as_of"], devices)
                update[tool] = data
                result = _summary(data)
        except FleetError as exc:
            trace.append({"event": "tool_refused", "tool": tool, "reason": str(exc)})
            status = "partial" if tool == "propose" else "error"
            update.update(status=status, notice=str(exc), attempted_draft=tool == "propose", trace=trace)
            return update
        except sqlite3.Error:
            trace.append({"event": "tool_error", "tool": tool, "reason": "storage_unavailable"})
            return {**update, "status": "error", "notice": "Telemetry or proposal storage is unavailable.",
                    "trace": trace}
        trace.append({"event": "tool_result", "tool": tool, "summary": result})
        return {**update, "trace": trace}

    def _finish(self, state: Turn):
        status = state.get("status", "complete")
        notice = state.get("notice")
        plan = state.get("plan")
        if status == "complete" and plan and plan.action and not state.get("proposals"):
            status = "partial"
            notice = "The available evidence supports no proposal for this request."
        return {"status": status, "notice": notice or "",
                "trace": _trace(state, "finished", status=status, notice=notice or "")}

    def ask(self, text: str, session: str = "default") -> dict:
        if not text.strip() or len(text) > 5000:
            raise FleetError("Enter a request between 1 and 5,000 characters.")
        memory, version = self.store.conversation(self.principal, session)
        with self.store.connect() as conn:
            as_of = self.store.as_of(conn, self.principal)
            devices = self.store.snapshots(conn, self.principal, as_of)
        if not as_of:
            raise FleetError("No telemetry is available for this account.")
        visible_ids = list(dict.fromkeys(row["device"] for row in devices))
        run_id = uuid4().hex
        state = self.graph.invoke({
            "run_id": run_id, "memory": memory, "text": text, "visible_ids": visible_ids,
            "trace": [], "tools_used": 0, "as_of": as_of,
        }, config={"recursion_limit": 2 * MAX_TOOLS + 8})
        status = state["status"]
        notice = state["notice"]
        primary = state.get("history") or state.get("latest")
        reference_group = memory.group
        if primary is not None and status in {"complete", "partial"}:
            plan = state["plan"]
            targets = state["devices"]
            group = primary["matches"] if targets is None else memory.group
            if targets is not None and not set(targets).issubset(group):
                group = targets
            updated = Conversation(group=group, focus=targets[0] if targets and len(targets) == 1 else None,
                                   query=plan.query)
            if self.store.save_conversation(self.principal, session, updated, version):
                reference_group = updated.group
            else:
                reference_group = self.store.conversation(self.principal, session)[0].group
                notice = (notice + " Another turn updated this session; this turn did not change its context.").strip()
        reports = []
        if primary is not None and state["plan"].query.metric in HANDLED_METRICS:
            query = state["plan"].query
            with self.store.connect() as conn:
                for item in primary["details"]:
                    reports.extend(view_handling(
                        self.store, conn, self.principal, item["device"], as_of, query.metric, query.check_id,
                    ))
        trace = state["trace"]
        self.store.log(self.principal, "agent_run", run_id,
                       {"planner": self.planner.name, "session": session, "trace": trace})
        if any(event["event"] == "tool_refused" and event["tool"] == "propose" for event in trace) or (
            state.get("plan") and state["plan"].action and status == "partial"
        ):
            self.store.log(self.principal, "proposal_refused", run_id, {"reason": notice})
        return {
            "run_id": run_id, "company": self.principal.company, "planner": self.planner.name,
            "status": status, "notice": notice, "as_of": as_of, "trace": trace,
            "reference_group": reference_group,
            "latest": state.get("latest"), "history": state.get("history"),
            "proposals": state.get("proposals", []),
            "handling_confirmation": state.get("handling_confirmation"),
            "handling_reports": reports,
        }
