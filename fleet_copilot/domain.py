"""Data exchanged at the planner, telemetry, and approval boundaries."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Metric = Literal["disk", "ram", "battery", "compliance", "os", "inventory"]
Mode = Literal["latest", "history", "latest_then_history"]
Tool = Literal["latest", "history", "propose", "finish"]
Severity = Literal["low", "medium", "high", "critical"]
Platform = Literal["darwin", "win32", "linux"]
DeviceKind = Literal["laptop", "desktop"]
Comparison = Literal["lt", "le", "gt", "ge"]
RefKind = Literal["fleet", "ids", "focus", "other", "group", "ordinal", "last"]
Intent = Literal["fleet", "help", "clarify", "decline"]
Action = Literal["ticket", "replacement_review"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Query(Contract):
    metric: Metric
    mode: Mode = "latest"
    days: int = Field(default=7, ge=2, le=30)
    platform: Platform | None = None
    device_kind: DeviceKind | None = None
    comparison: Comparison | None = None
    threshold_percent: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    severity: Severity | None = None
    check_id: str | None = Field(default=None, max_length=80)
    os_before: str | None = Field(default=None, pattern=r"^\d{1,5}(\.\d{1,5}){0,3}$")
    os_version: str | None = Field(default=None, pattern=r"^\d{1,5}(\.\d{1,5}){0,3}$")

    @model_validator(mode="after")
    def validate_parameters(self):
        if self.metric != "compliance" and (self.severity or self.check_id):
            raise ValueError("Severity and check ID apply to compliance queries")
        if (self.os_before or self.os_version) and self.platform is None:
            raise ValueError("An OS version constraint needs a platform")
        if self.os_before and self.os_version:
            raise ValueError("Choose one OS version constraint")
        if self.comparison is not None and self.threshold_percent is None:
            raise ValueError("A comparison requires an explicit threshold")
        if self.metric not in {"disk", "ram"} and self.threshold_percent is not None:
            raise ValueError("A utilization threshold applies to disk or RAM")
        if self.mode != "latest" and self.metric in {"os", "inventory"}:
            raise ValueError("OS and inventory are available as latest-state queries")
        return self


class Plan(Contract):
    intent: Intent = "fleet"
    query: Query | None = None
    reference: RefKind = "fleet"
    device_ids: list[str] = Field(default_factory=list, max_length=40)
    ordinal: int | None = Field(default=None, ge=1, le=40)
    company: str | None = Field(default=None, max_length=80)
    action: Action | None = None
    clarification: Literal["metric", "device", "time_window", "request"] = "request"

    @model_validator(mode="after")
    def validate_plan(self):
        if self.intent == "fleet" and self.query is None:
            raise ValueError("A fleet plan needs a query")
        if self.action is not None and self.intent != "fleet":
            raise ValueError("Only a fleet plan may request an action")
        if self.reference == "ids" and not self.device_ids:
            raise ValueError("An explicit device reference needs IDs")
        if self.reference != "ids" and self.device_ids:
            raise ValueError("Only an explicit reference can carry device IDs")
        if (self.reference == "ordinal") != (self.ordinal is not None):
            raise ValueError("An ordinal reference needs a position")
        if self.action == "replacement_review" and (not self.query or self.query.metric != "battery"):
            raise ValueError("Replacement review requires battery evidence")
        return self


class Decision(Contract):
    tool: Tool


class Conversation(Contract):
    group: list[str] = Field(default_factory=list)
    focus: str | None = None
    query: Query | None = None


@dataclass(frozen=True)
class Principal:
    company: str
    actor: str
    role: str


class FleetError(Exception):
    """An expected error safe to show in the CLI."""


APPROVAL_HINT = "Approve or reject through the proposal command using the displayed proposal ID."
