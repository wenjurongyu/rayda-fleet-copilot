"""Deterministic development checks, with expected values from synthetic cases."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
import os
import sqlite3
import sys
from types import ModuleType

import pytest

from fleet_copilot.actions import review
from fleet_copilot.agent import Agent
from fleet_copilot.domain import Decision, FleetError
from fleet_copilot.planner import ModelPlanner, OfflinePlanner
from fleet_copilot.store import Store
from scripts.make_demo import ACME, END, OTHER, demo_rows, write_dataset


@pytest.fixture
def fleet(tmp_path):
    path = tmp_path / "demo.ndjson"
    write_dataset(path)
    store = Store(tmp_path / "fleet.db")
    assert store.ingest(path)["inserted"] == 23
    assert store.ingest(path)["duplicates"] == 23
    admin = store.authenticate(store.add_principal(ACME, "admin", "admin"))
    other = store.authenticate(store.add_principal(OTHER, "admin", "admin"))
    viewer = store.authenticate(store.add_principal(ACME, "viewer", "viewer"))
    return store, admin, other, viewer


def calls(report):
    return [event for event in report["trace"] if event["event"] == "tool_call"]


def test_agent_changes_path_after_observation(fleet):
    store, admin, *_ = fleet
    agent = Agent(store, admin, OfflinePlanner())
    question = "Which currently low disk devices were persistently low over the last 7 days?"
    result = agent.ask(question)
    assert result["latest"]["matches"] == ["DEMO-A", "DEMO-B"]
    assert result["history"]["matches"] == ["DEMO-A"]
    assert result["history"]["unknown"] == ["DEMO-B"]
    assert [event["tool"] for event in calls(result)] == ["latest", "history"]
    assert calls(result)[1]["arguments"]["devices"] == ["DEMO-A", "DEMO-B"]
    none = agent.ask(question + " Use less than 1%.")
    assert none["latest"]["matches"] == []
    assert none["history"] is None
    assert [event["tool"] for event in calls(none)] == ["latest"]


def test_followup_resolves_other_device_and_inherits_query(fleet):
    store, admin, *_ = fleet
    agent = Agent(store, admin, OfflinePlanner())
    agent.ask("Which devices have low disk space?", "conversation")
    first = agent.ask('Show the disk history for device "DEMO-A" over the past 7 days', "conversation")
    second = agent.ask("What about the other one?", "conversation")
    assert first["history"]["details"][0]["observed_days"] == 7
    assert second["history"]["unknown"] == ["DEMO-B"]
    assert calls(second)[0]["arguments"]["devices"] == ["DEMO-B"]
    assert calls(second)[0]["arguments"]["query"]["metric"] == "disk"
    assert calls(second)[0]["arguments"]["query"]["days"] == 7
    unrelated = agent.ask("What about the other one?", "new-conversation")
    assert unrelated["status"] == "clarify"
    assert calls(unrelated) == []


def test_explicit_ordinal_and_current_scope(fleet):
    store, admin, *_ = fleet
    agent = Agent(store, admin, OfflinePlanner())
    agent.ask("Which devices have low disk space?")
    report = agent.ask("What about the second one over the past 7 days?")
    assert calls(report)[0]["arguments"]["devices"] == ["DEMO-B"]


def test_traceable_empty_compliance_result(fleet):
    store, admin, *_ = fleet
    result = Agent(store, admin, OfflinePlanner()).ask("Which laptops fail high-severity compliance checks?")
    assert result["latest"]["matches"] == []
    assert result["latest"]["unknown"] == []
    assert {entry["device"] for entry in result["latest"]["details"]} == {"DEMO-A", "DEMO-B"}
    citation = result["latest"]["details"][0]["evidence"][0]
    snapshot = store.evidence(admin, **citation)
    assert {"check_id": "disk_encryption", "status": "pass", "severity": "high"} in snapshot["checks"]


def test_compliance_drift_and_os_version_scope(fleet):
    store, admin, *_ = fleet
    agent = Agent(store, admin, OfflinePlanner())
    drift = agent.ask("Show screen lock compliance drift for the past 7 days")
    a = next(device for device in drift["history"]["details"] if device["device"] == "DEMO-A")
    assert a["new_failures"] == ["screen_lock"]
    assert a["breach_days"] == 3
    os = agent.ask("How many devices are running an OS older than macOS 15?")
    assert os["latest"]["matches"] == ["DEMO-A"]


def test_tenant_isolation_for_queries_evidence_and_conversations(fleet):
    store, admin, other, _ = fleet
    agent = Agent(store, admin, OfflinePlanner())
    outside = agent.ask("Show the disk status for Globex")
    assert outside["status"] == "declined" and calls(outside) == []
    outside = agent.ask('Show the disk status for device "DEMO-OTHER"')
    assert outside["status"] == "declined" and calls(outside) == []
    own = Agent(store, other, OfflinePlanner()).ask("Which devices have low disk space?")
    reference = own["latest"]["details"][0]["evidence"][0]
    with pytest.raises(FleetError, match="unavailable"):
        store.evidence(admin, **reference)
    assert store.audit(admin, own["run_id"]) == []
    isolated = agent.ask("What about that device?")
    assert isolated["status"] == "clarify"


def _ticket(store, admin):
    result = Agent(store, admin, OfflinePlanner()).ask("Draft a ticket for failing screen_lock compliance checks")
    proposal, = result["proposals"]
    assert result["status"] == "complete" and proposal["status"] == "pending"
    assert proposal["payload"]["device"] == "DEMO-A"
    assert proposal["payload"]["finding"]["check_id"] == "screen_lock"
    return proposal["id"]


def test_approval_separates_pending_review_and_effect(fleet):
    store, admin, other, viewer = fleet
    proposal = _ticket(store, admin)
    assert store.get_proposal(admin, proposal)["simulation"] is None
    for principal in (viewer, other):
        with pytest.raises(FleetError, match="unavailable"):
            review(store, principal, proposal, True)
    response = Agent(store, admin, OfflinePlanner()).ask("Approve the ticket now")
    assert response["status"] == "declined"
    assert store.get_proposal(admin, proposal)["simulation"] is None
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(lambda _: review(store, admin, proposal, True), range(2)))
    assert all(reply["status"] == "executed" for reply in replies)
    assert replies[0]["simulation"] == replies[1]["simulation"]
    events = store.audit(admin, proposal)
    assert sum(event["event"] == "simulation_executed" for event in events) == 1
    assert sum(event["event"] == "proposal_approved" for event in events) == 1


def test_rejection_is_terminal(fleet):
    store, admin, *_ = fleet
    proposal = _ticket(store, admin)
    assert review(store, admin, proposal, False)["status"] == "rejected"
    later = review(store, admin, proposal, True)
    assert later["status"] == "rejected" and later["simulation"] is None


def test_new_snapshot_invalidates_old_approval(fleet, tmp_path):
    store, admin, *_ = fleet
    proposal = _ticket(store, admin)
    path = tmp_path / "new.ndjson"
    row = [entry for entry in demo_rows() if entry["device_id"] == "DEMO-A"][-1]
    row["collected_at"] = (END + timedelta(days=1)).isoformat() + "T12:00:00Z"
    row["compliance_results"][0]["status"] = "pass"
    write_dataset(path, [row])
    store.ingest(path)
    result = review(store, admin, proposal, True)
    assert result["status"] == "expired" and result["simulation"] is None


def test_missing_battery_and_weak_threshold_cannot_support_action(fleet):
    store, admin, *_ = fleet
    agent = Agent(store, admin, OfflinePlanner())
    battery = agent.ask('Propose a replacement review for battery on device "DEMO-C"')
    assert battery["status"] == "partial"
    assert battery["latest"]["unknown"] == ["DEMO-C"]
    assert battery["proposals"] == []
    disk = agent.ask("Draft a ticket for disk usage less than 99% free over the past 7 days")
    assert disk["status"] == "partial"
    assert disk["proposals"] == []
    assert any(event["event"] == "tool_refused" and event["tool"] == "propose" for event in disk["trace"])


def test_policy_supports_sustained_disk_and_battery_review(fleet):
    store, admin, *_ = fleet
    agent = Agent(store, admin, OfflinePlanner())
    disk = agent.ask("Draft a ticket for devices currently low on disk and persistently low over the past 7 days")
    proposal, = disk["proposals"]
    assert proposal["payload"]["device"] == "DEMO-A"
    assert proposal["payload"]["finding"]["observed_days"] == 7
    assert len(proposal["payload"]["history"]) == 7
    battery = agent.ask('Propose a battery replacement review for device "DEMO-A"')
    proposal, = battery["proposals"]
    assert proposal["payload"]["action"] == "replacement_review"
    assert review(store, admin, proposal["id"], True)["status"] == "executed"


def test_model_tool_selection_is_validated_against_graph(fleet):
    class OutOfBoundsPlanner(OfflinePlanner):
        def next(self, plan, offered, observations):
            return Decision(tool="propose")

    store, admin, *_ = fleet
    agent = Agent(store, admin, OutOfBoundsPlanner())
    report = agent.ask("Show disk trends")
    assert report["status"] == "error"
    assert calls(report) == []
    assert report["proposals"] == []


def test_database_tool_error_is_distinct_from_no_matches(fleet, monkeypatch):
    from fleet_copilot import agent

    def unavailable(*_args, **_kwargs):
        raise sqlite3.OperationalError("test storage failure")

    store, admin, *_ = fleet
    monkeypatch.setattr(agent, "read_latest", unavailable)
    report = Agent(store, admin, OfflinePlanner()).ask("Show low disk devices")
    assert report["status"] == "error"
    assert report["latest"] is None
    assert any(event["event"] == "tool_error" for event in report["trace"])


def test_ambiguous_other_reference_stops_before_tool_call(fleet):
    store, admin, *_ = fleet
    agent = Agent(store, admin, OfflinePlanner())
    group = agent.ask("List devices")
    assert group["latest"]["matches"] == ["DEMO-A", "DEMO-B", "DEMO-C"]
    agent.ask("Show disk status for DEMO-A")
    result = agent.ask("What about the other one?")
    assert result["status"] == "clarify"
    assert "DEMO-B" in result["notice"] and "DEMO-C" in result["notice"]
    assert calls(result) == []


def test_explicit_comparison_direction_and_action_threshold(fleet):
    store, admin, *_ = fleet
    agent = Agent(store, admin, OfflinePlanner())
    high = agent.ask("How many devices have more than 95% free disk space?")
    assert high["latest"]["matches"] == []
    assert high["latest"]["criterion"] == "system volume free space > 95%"
    exact = agent.ask("How many devices have at least 60% free disk space?")
    assert exact["latest"]["matches"] == ["DEMO-C"]
    above = agent.ask("Draft a disk pressure ticket for DEMO-A only if it has more than 5% free disk space.")
    assert above["latest"]["matches"] == ["DEMO-A"]
    assert above["status"] == "partial" and above["proposals"] == []


@pytest.mark.parametrize(("question", "matching"), [
    ("How many devices have 7% or more free disk space?", ["DEMO-B", "DEMO-C"]),
    ("Show devices with 6 percent or less free disk space.", ["DEMO-A"]),
    ("List devices at or above 60% free disk.", ["DEMO-C"]),
    ("Show low disk devices with a 7% threshold.", ["DEMO-A"]),
])
def test_percent_comparison_paraphrases(fleet, question, matching):
    store, admin, *_ = fleet
    response = Agent(store, admin, OfflinePlanner()).ask(question)
    assert response["latest"]["matches"] == matching


@pytest.mark.parametrize("condition", [
    "if it currently has 7% or more free disk space",
    "only if it has 7% or more free disk space",
    "only if it has 7% or less free disk space by Friday",
    "only if it has below 7% free disk and also has latest macOS",
])
def test_draft_conditions_are_checked_as_written(fleet, condition):
    store, admin, *_ = fleet
    response = Agent(store, admin, OfflinePlanner()).ask(f"Draft a disk pressure ticket for DEMO-A {condition}.")
    assert response["proposals"] == []
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] == 0


def test_supported_explicit_condition_and_combined_request_can_draft(fleet):
    store, admin, *_ = fleet
    agent = Agent(store, admin, OfflinePlanner())
    condition = agent.ask("Draft a disk pressure ticket for DEMO-A only if it has 6% or less free disk space.")
    assert [item["payload"]["device"] for item in condition["proposals"]] == ["DEMO-A"]
    combined = agent.ask("Find failing screen_lock compliance checks and draft a ticket.")
    assert [item["payload"]["device"] for item in combined["proposals"]] == ["DEMO-A"]


@pytest.mark.parametrize("question", [
    "I do not want you to draft a ticket for failing screen_lock compliance checks on DEMO-A.",
    "Show failing screen_lock compliance checks on DEMO-A without opening a ticket.",
    "How should I draft a ticket for failing screen_lock compliance checks?",
    'Show the text "draft a ticket for failing screen_lock compliance checks."',
])
def test_no_proposals_without_an_explicit_affirmative_request(fleet, question):
    store, admin, *_ = fleet
    result = Agent(store, admin, OfflinePlanner()).ask(question)
    assert result["proposals"] == []
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] == 0


def test_planner_cannot_silently_reverse_explicit_comparison(fleet):
    class ReversedPlanner(OfflinePlanner):
        def plan(self, text, memory, device_ids):
            plan = super().plan(text, memory, device_ids)
            return plan.model_copy(update={"query": plan.query.model_copy(update={"comparison": "lt"})})

    store, admin, *_ = fleet
    result = Agent(store, admin, ReversedPlanner()).ask("How many devices have more than 95% free disk space?")
    assert result["status"] == "clarify" and calls(result) == []


def test_negated_ticket_request_remains_read_only(fleet):
    store, admin, *_ = fleet
    agent = Agent(store, admin, OfflinePlanner())
    result = agent.ask("Do not draft a ticket for failing screen_lock compliance checks on device DEMO-A.")
    assert result["latest"]["matches"] == ["DEMO-A"] and result["proposals"] == []
    assert any(event["event"] == "action_suppressed" for event in result["trace"])
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM proposals WHERE company=?", (admin.company,)).fetchone()[0] == 0


def test_unknown_explicit_device_cannot_widen_action(fleet):
    store, admin, *_ = fleet
    agent = Agent(store, admin, OfflinePlanner())
    result = agent.ask("Draft a ticket for failing screen_lock compliance checks on device qa-ghost-42.")
    assert result["status"] == "declined"
    assert calls(result) == [] and result["proposals"] == []


@pytest.mark.parametrize("question", [
    "Draft a ticket for failing screen_lock compliance checks for devices located in London.",
    "Draft disk pressure tickets for devices with less than 5 GB free disk space.",
    "Draft a disk pressure ticket for DEMO-A only if it has less than 5% used disk space.",
    "Draft a disk pressure ticket for DEMO-A only if used disk space falls below 5%.",
    "Draft a ticket for failing screen_lock compliance checks on devices running macOS newer than 15.4.",
    "Draft a screen_lock compliance ticket for devices whose macOS version is newer than 15.4.",
])
def test_unsupported_filters_remain_read_only(fleet, question):
    store, admin, *_ = fleet
    result = Agent(store, admin, OfflinePlanner()).ask(question)
    assert result["proposals"] == []
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] == 0


def test_inconsistent_planner_action_returns_controlled_error(fleet):
    from fleet_copilot.domain import Plan

    class InconsistentPlanner(OfflinePlanner):
        def plan(self, text, memory, device_ids):
            return Plan.model_construct(intent="help", action="ticket")

    store, admin, *_ = fleet
    result = Agent(store, admin, InconsistentPlanner()).ask("Help")
    assert result["status"] == "error" and calls(result) == []
    assert result["proposals"] == []


def test_exact_os_version_filters_proposal_and_evidence(fleet, tmp_path):
    store, admin, *_ = fleet
    latest = [item for item in demo_rows() if item["device_id"] == "DEMO-B"][-1]
    latest["collected_at"] = (END + timedelta(days=1)).isoformat() + "T12:00:00Z"
    latest["compliance_results"][0]["status"] = "fail"
    path = tmp_path / "second-failure.ndjson"
    write_dataset(path, [latest])
    store.ingest(path)
    agent = Agent(store, admin, OfflinePlanner())
    report = agent.ask("Draft a ticket for failing screen_lock compliance checks only on devices running macOS 15.1.")
    proposal, = report["proposals"]
    assert proposal["payload"]["device"] == "DEMO-B"
    assert proposal["payload"]["query"]["os_version"] == "15.1"
    assert report["latest"]["scope"]["os_version"] == "15.1"
    assert review(store, admin, proposal["id"], True)["status"] == "executed"


def test_displayed_ordinal_is_the_one_the_next_turn_uses(fleet, capsys):
    from fleet_copilot.cli import _render

    store, admin, *_ = fleet
    agent = Agent(store, admin, OfflinePlanner())
    listing = agent.ask("Which devices have more than 6% free disk space?", "order")
    assert listing["reference_group"] == ["DEMO-B", "DEMO-C"]
    _render(listing, False)
    display = capsys.readouterr().out
    assert "Follow-up device order\n  1. DEMO-B\n  2. DEMO-C" in display
    assert display.index("  DEMO-B:") < display.index("  DEMO-A:")
    action = agent.ask("Draft a screen_lock compliance ticket for the first device listed.", "order")
    assert calls(action)[0]["arguments"]["devices"] == ["DEMO-B"]
    assert action["proposals"] == []


@pytest.mark.parametrize("anchor", ["on 2026-06-09", "in 2025", "as of 09:00"])
def test_dated_compliance_request_cannot_use_current_failures(fleet, anchor):
    store, admin, *_ = fleet
    result = Agent(store, admin, OfflinePlanner()).ask(
        f"Draft a ticket for failing screen_lock compliance checks {anchor}."
    )
    assert result["status"] == "clarify" and calls(result) == []
    assert result["proposals"] == []


def test_last_device_never_expands_to_the_fleet(fleet):
    store, admin, *_ = fleet
    agent = Agent(store, admin, OfflinePlanner())
    listed = agent.ask("Which devices have low disk space?", "ordered")
    assert listed["reference_group"] == ["DEMO-A", "DEMO-B"]
    request = "Draft a screen_lock compliance ticket only for the last device listed."
    response = agent.ask(request, "ordered")
    assert calls(response)[0]["arguments"]["devices"] == ["DEMO-B"]
    assert response["proposals"] == []
    fresh = agent.ask(request, "new-session")
    assert fresh["status"] == "clarify" and not calls(fresh)


def test_contradictory_compliance_measurements_support_no_ticket(fleet, tmp_path):
    store, admin, *_ = fleet
    latest = [item for item in demo_rows() if item["device_id"] == "DEMO-A"][-1]
    latest["collected_at"] = (END + timedelta(days=1)).isoformat() + "T12:00:00Z"
    latest["compliance_results"] = [
        {"check_id": "screen_lock", "status": "pass", "severity": "low"},
        {"check_id": "screen_lock", "status": "fail", "severity": "high"},
    ]
    path = tmp_path / "ambiguous.ndjson"
    write_dataset(path, [latest])
    store.ingest(path)
    result = Agent(store, admin, OfflinePlanner()).ask("Draft a ticket for failing screen_lock checks on DEMO-A.")
    assert result["latest"]["unknown"] == ["DEMO-A"]
    assert result["latest"]["details"][0]["data"]["ambiguous_checks"] == ["screen_lock"]
    assert result["proposals"] == []


def test_repeated_request_reuses_pending_proposal(fleet):
    store, admin, *_ = fleet
    original = _ticket(store, admin)
    repeat = _ticket(store, admin)
    assert original == repeat
    assert [event["event"] for event in store.audit(admin, original)] == ["proposal_created", "proposal_reused"]


def test_unknown_person_filter_does_not_expand_proposal_scope(fleet):
    store, admin, *_ = fleet
    result = Agent(store, admin, OfflinePlanner()).ask(
        "Draft a ticket for failing screen_lock checks only for devices assigned to Alex."
    )
    assert result["proposals"] == []
    assert any(event.get("reason") == "unsupported_action_condition" for event in result["trace"])


def test_laptop_filter_excludes_desktop_and_reports_unknown(fleet, tmp_path):
    store, admin, *_ = fleet
    laptop = [item for item in demo_rows() if item["device_id"] == "DEMO-A"][-1]
    desktop = [item for item in demo_rows() if item["device_id"] == "DEMO-C"][-1]
    unknown = {**desktop, "device_id": "DEMO-UNKNOWN", "device_identity": {}}
    for row in (laptop, desktop, unknown):
        row["collected_at"] = (END + timedelta(days=1)).isoformat() + "T12:00:00Z"
        row["compliance_results"] = [{"check_id": "disk_encryption", "status": "fail", "severity": "high"}]
    path = tmp_path / "new.ndjson"
    write_dataset(path, [laptop, desktop, unknown])
    store.ingest(path)
    result = Agent(store, admin, OfflinePlanner()).ask("Show me laptops failing high-severity compliance checks.")
    assert result["latest"]["matches"] == ["DEMO-A"]
    assert [item["device"] for item in result["latest"]["unclassified"]] == ["DEMO-UNKNOWN"]


@pytest.mark.parametrize(("provider", "credential"), [
    ("anthropic", "ANTHROPIC_API_KEY"), ("google_genai", "GOOGLE_API_KEY"),
])
def test_native_provider_uses_shared_typed_planner_contract(monkeypatch, provider, credential):
    from fleet_copilot.domain import Conversation, Plan

    class FakeChatModel:
        def with_structured_output(self, contract):
            assert contract is Plan
            return self

        def invoke(self, messages):
            assert messages[0]["role"] == "system"
            return Plan(intent="help")

    def initialize(model, *, model_provider, timeout, max_retries, api_key, base_url, **kwargs):
        assert (model, model_provider, timeout, max_retries) == ("fixture-model", provider, 30, 1)
        assert api_key == "fixture-key"
        if provider == "anthropic":
            assert base_url == "https://api.anthropic.com" and kwargs == {}
        else:
            assert base_url == "https://generativelanguage.googleapis.com" and kwargs == {"vertexai": False}
        return FakeChatModel()

    module = ModuleType("langchain.chat_models")
    module.init_chat_model = initialize
    monkeypatch.setitem(sys.modules, "langchain.chat_models", module)
    monkeypatch.setenv("RAYDA_PROVIDER", provider)
    monkeypatch.setenv("RAYDA_MODEL", "fixture-model")
    monkeypatch.setenv(credential, "fixture-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://untrusted.invalid")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "http://untrusted.invalid")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "false")
    assert ModelPlanner().plan("Hello", Conversation(), []).intent == "help"


@pytest.mark.parametrize("flag", ["GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE"])
def test_gemini_native_profile_refuses_conflicting_ambient_backend(monkeypatch, flag):
    from fleet_copilot.model_config import ModelSettings

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.setenv(flag, "true")
    settings = ModelSettings(provider="google_genai", model="fixture-model")
    with pytest.raises(FleetError, match=flag):
        ModelPlanner(settings)


def test_model_serialization_cannot_replace_an_explicit_target():
    from fleet_copilot.domain import Plan
    from fleet_copilot.planner import _validate_model_json

    candidate = """{"intent":"fleet","reference":"focus","device_ids":["DEMO-B"],
                    "query":{"metric":"compliance","check_id":"screen_lock","action":"ticket"},
                    "action":"ticket"}"""
    plan = _validate_model_json(Plan, candidate, {"explicit_device_ids": ["DEMO-A"]})
    assert plan.reference == "ids" and plan.device_ids == ["DEMO-A"]
    assert plan.query.check_id == "screen_lock" and plan.action == "ticket"


def test_model_action_fields_in_query_preserve_scope_and_detect_conflicts():
    import json
    from fleet_copilot.domain import Plan
    from fleet_copilot.planner import _validate_model_json

    raw = {
        "query": {"metric": "compliance", "check_id": "screen_lock", "company": "acme-001",
                  "action": "ticket", "reference": "ids", "device_ids": ["DEMO-A"]},
        "company": None,
    }
    plan = _validate_model_json(Plan, json.dumps(raw), {})
    assert (plan.company, plan.action, plan.device_ids) == ("acme-001", "ticket", ["DEMO-A"])
    raw["company"] = "globex-002"
    with pytest.raises(ValueError, match="Conflicting"):
        _validate_model_json(Plan, json.dumps(raw), {})


def test_custom_model_endpoint_requires_its_own_explicit_credential(monkeypatch):
    monkeypatch.setenv("RAYDA_PROVIDER", "openai-compatible")
    monkeypatch.setenv("RAYDA_MODEL", "fixture-model")
    monkeypatch.setenv("RAYDA_BASE_URL", "https://example.invalid/v1")
    monkeypatch.delenv("RAYDA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    with pytest.raises(FleetError, match="explicitly"):
        ModelPlanner()


@pytest.mark.skipif(not os.getenv("FLEET_ASSESSMENT_DATA"), reason="set FLEET_ASSESSMENT_DATA to the assessment NDJSON")
def test_official_dataset_public_expectations(tmp_path):
    store = Store(tmp_path / "official.db")
    manifest = store.ingest(Path(os.environ["FLEET_ASSESSMENT_DATA"]))
    assert manifest["inserted"] == 750
    assert len(manifest["companies"]) == 3
    assert sum(item["devices"] for item in manifest["companies"]) == 25
    for item in manifest["companies"]:
        principal = store.authenticate(store.add_principal(item["company"], "admin", "admin"))
        agent = Agent(store, principal, OfflinePlanner())
        high = agent.ask("Which devices fail high-severity compliance checks?")
        assert high["latest"]["matches"] == high["latest"]["unknown"] == []
        low_disk = agent.ask("Which devices are currently low on disk and persistently low over the past 7 days?")
        assert len(low_disk["latest"]["matches"]) == len(low_disk["history"]["matches"]) == 2



def test_spoken_handling_requires_explicit_admin_confirmation(fleet):
    from fleet_copilot.handling import list_handling

    store, admin, other, viewer = fleet
    agent = Agent(store, admin, OfflinePlanner())
    agent.ask("Show the latest disk status for DEMO-A.", "handling")
    for question in ("I've already fixed it.", "我已经处理过这台机器了"):
        report = agent.ask(question, "handling")
        assert report["status"] == "confirm_handling"
        assert report["handling_confirmation"] == {"device": "DEMO-A", "metric": "disk", "check_id": None}
        assert report["latest"] is None
    assert list_handling(store, admin, "DEMO-A") == []
    assert agent.ask("I fixed DEMO-A's battery, not its disk.", "handling")["status"] == "clarify"
    assert agent.ask("I fixed another device.", "handling")["status"] == "clarify"
    assert Agent(store, viewer, OfflinePlanner()).ask("I fixed DEMO-A's disk.")["status"] == "declined"
    assert Agent(store, other, OfflinePlanner()).ask("I fixed the disk on device DEMO-A.")["status"] == "clarify"


def test_reconciliation_uses_collection_time_and_keeps_original_evidence(fleet, tmp_path):
    from fleet_copilot.handling import list_handling, record_handling

    store, admin, *_ = fleet
    agent = Agent(store, admin, OfflinePlanner())
    before = agent.ask("Show the disk status for DEMO-A.")["latest"]["details"][0]
    record = record_handling(store, admin, "DEMO-A", "disk", None, END.isoformat() + "T15:00:00Z")
    assert record["status"] == "pending_verification"
    assert record["baseline"] == before["evidence"][0]["collected"]
    assert record_handling(store, admin, "DEMO-A", "disk", None, END.isoformat() + "T15:00:00Z")["id"] == record["id"]
    assert [event["event"] for event in store.audit(admin, record["id"])] == ["handling_reported"]

    path = tmp_path / "out-of-order.ndjson"
    base = [entry for entry in demo_rows() if entry["device_id"] == "DEMO-A"][-1]

    def ingest_at(hour, free):
        row = {**base, "collected_at": END.isoformat() + f"T{hour:02d}:00:00Z"}
        row["disk_volumes"] = [{**base["disk_volumes"][0], "available_bytes": free}]
        write_dataset(path, [row])
        store.ingest(path)

    ingest_at(14, 80)
    assert list_handling(store, admin, "DEMO-A")[0]["status"] == "pending_verification"
    report = agent.ask("Show the disk status for DEMO-A.")
    assert report["handling_reports"][0]["id"] == record["id"]
    assert report["latest"]["details"][0]["data"]["free_percent"] == 80

    ingest_at(16, 6)
    assert list_handling(store, admin, "DEMO-A")[0]["status"] == "still_abnormal"
    ingest_at(18, 72)
    ingest_at(17, 6)  # Received after the newer successful verification.
    result = list_handling(store, admin, "DEMO-A")[0]
    assert result["status"] == "verified"
    assert result["verification"]["measurement"]["free_percent"] == 72
    assert "T18:00:00" in result["verification"]["collected"]
    original = store.evidence(admin, **before["evidence"][0])
    assert original["volumes"][0]["available_bytes"] == 6


def test_proposal_hold_is_issue_scoped_and_blocks_approval(fleet, tmp_path):
    from fleet_copilot.handling import record_handling

    store, admin, other, viewer = fleet
    ticket = _ticket(store, admin)
    with pytest.raises(FleetError, match="administrator"):
        record_handling(store, viewer, "DEMO-A", "compliance", "screen_lock")
    with pytest.raises(FleetError, match="unavailable"):
        record_handling(store, other, "DEMO-A", "compliance", "screen_lock")
    report = record_handling(store, admin, "DEMO-A", "compliance", "screen_lock", END.isoformat() + "T13:00:00Z")
    assert report["status"] == "pending_verification"
    same_request = Agent(store, admin, OfflinePlanner()).ask("Draft a ticket for failing screen_lock compliance checks")
    assert same_request["status"] == "partial" and same_request["proposals"] == []
    assert review(store, admin, ticket, True)["status"] == "expired"
    disk = Agent(store, admin, OfflinePlanner()).ask("Draft a ticket for devices currently low on disk and persistently low over the past 7 days")
    assert [record["payload"]["device"] for record in disk["proposals"]] == ["DEMO-A"]

    # A fresh abnormal measurement can support a new proposal for the original issue.
    row = [item for item in demo_rows() if item["device_id"] == "DEMO-A"][-1]
    row["collected_at"] = END.isoformat() + "T14:00:00Z"
    path = tmp_path / "fresh.ndjson"
    write_dataset(path, [row])
    store.ingest(path)
    assert _ticket(store, admin) != ticket


def test_model_toml_matches_profile_and_keeps_credentials_in_environment(tmp_path, monkeypatch):
    from fleet_copilot.model_config import load_model_settings

    monkeypatch.setenv("OPENAI_API_KEY", "fixture-openai")
    monkeypatch.setenv("RAYDA_MODEL", "legacy-model")
    monkeypatch.delenv("MIMO_TEST_KEY", raising=False)
    path = tmp_path / "models.toml"
    path.write_text(
        'default_profile = "cloud"\n[profiles.cloud]\nprovider = "openai-compatible"\n'
        'base_url = "https://example.invalid/v1"\nmodel = "configured-model"\napi_key_env = "MIMO_TEST_KEY"\n'
        'timeout_seconds = 55\n[profiles.local]\nmodel = "fixture-local"\n'
        'base_url = "http://127.0.0.1:18478/v1"\napi_key_env = "LOCAL_TEST_KEY"\n',
        encoding="utf-8",
    )
    chosen = load_model_settings(path)
    assert (chosen.model, chosen.timeout_seconds) == ("configured-model", 55)
    with pytest.raises(FleetError, match="MIMO_TEST_KEY"):
        chosen.credential()
    monkeypatch.setenv("MIMO_TEST_KEY", "fixture-mimo")
    assert chosen.credential() == "fixture-mimo"
    assert load_model_settings(path, "local").model == "fixture-local"
    with pytest.raises(FleetError, match="profile"):
        load_model_settings(path, "unknown")
    path.write_text(path.read_text(encoding="utf-8").replace("model = \"configured-model\"", 'model = "configured-model"\napi_key = "fixture-secret"'), encoding="utf-8")
    with pytest.raises(FleetError, match="api_key") as denied:
        load_model_settings(path)
    assert "fixture-secret" not in str(denied.value)



def test_deepseek_profile_requests_bounded_json(monkeypatch):
    import json
    from types import SimpleNamespace

    from fleet_copilot import planner as planner_module
    from fleet_copilot.domain import Conversation
    from fleet_copilot.model_config import load_model_settings

    expected_url = "https://api.deepseek.com"

    def completion(**request):
        assert request["model"] == "deepseek-flash"
        assert request["response_format"] == {"type": "json_object"}
        assert json.loads(request["messages"][-1]["content"])["output_format"] == "json"
        assert request["extra_body"] == {"thinking": {"type": "disabled"}}
        assert request["max_tokens"] == 768
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"intent":"help"}'))])

    def sdk(*, api_key, base_url, **options):
        assert api_key == "fixture-deepseek-key" and base_url == expected_url
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-deepseek-key")
    monkeypatch.setattr(planner_module, "OpenAI", sdk)
    path = Path(__file__).resolve().parents[1] / "config" / "models.example.toml"
    model = ModelPlanner(load_model_settings(path, "deepseek"))
    assert model.plan("Hello", Conversation(), []).intent == "help"
