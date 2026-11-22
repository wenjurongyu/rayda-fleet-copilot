"""Interpret explicit conditions before any planner can draft an action."""

import re
import unicodedata
from dataclasses import dataclass

from .domain import Action, Comparison, Platform

_LABELLED_DEVICE = re.compile(
    r"\b(?:device|laptop|machine)\s+(?:id\s*[:=]?\s*)?"
    r"(?:\"(?P<double>[a-z0-9][\w.-]*)\"|'(?P<single>[a-z0-9][\w.-]*)'"
    r"|(?P<bare>[a-z0-9][\w.-]*))", re.IGNORECASE
)
_TARGET_STOP = frozenset({
    "a", "and", "are", "by", "failing", "group", "has", "health", "identity", "ids", "in", "is",
    "it", "listed", "management", "model", "named", "on", "order", "owner", "running", "status",
    "telemetry", "that", "the", "with", "without",
})
_OS_NAME = re.compile(r"\b(?P<family>macos|windows|linux)\b(?:\s+(?:version\s+)?(?P<version>\d+(?:\.\d+){0,3}))?")
_OS_PLATFORM: dict[str, Platform] = {"macos": "darwin", "windows": "win32", "linux": "linux"}
_OS_BEFORE = re.compile(r"(?:older than|before|earlier than|below|<)\s*$")
_OS_UNSUPPORTED_PREFIX = re.compile(r"(?:newer than|after|later than|at least|>=|>|except|exclude|excluding|not)\s*$")
_OS_RELATIVE_AFTER = re.compile(
    r"\s+(?:version\s+)?(?:is\s+)?(?P<direction>older than|earlier than|before|below|newer than|later than|after|"
    r"above|at least|at most|greater than|less than|>=|<=|>|<|equal to)\s+"
    r"(?P<version>\d+(?:\.\d+){0,3})"
)
_OS_UNSUPPORTED_SUFFIX = re.compile(r"\s*(?:or|and)\s+(?:later|earlier|older|newer|above|below|above it|up)\b")
_PERCENT = re.compile(r"(?<![\d.\-+])(-?\d+(?:\.\d+)?)\s*(?:%|percent\b)")
_PREFIX = {
    "ge": r">=|≥|at least|no less than|not less than|at or above|greater than or equal to|不少于|不低于|至少|大于等于",
    "le": r"<=|≤|at most|no more than|not more than|at or below|less than or equal to|不超过|不高于|至多|小于等于",
    "gt": r">|more than|greater than|higher than|above|over|超过|大于|高于",
    "lt": r"<|under|below|less than|fewer than|lower than|低于|小于|不到",
}
_SUFFIX = {
    "ge": r"(?:or|and)\s+(?:more|above|higher|greater)\b|\+|及以上|以上",
    "le": r"(?:or|and)\s+(?:less|below|lower|fewer|under)\b|及以下|以下",
}
_ACTION = r"(?:draft|propose|open|create|file|raise|prepare|generate|make|flag|put in)"
_POSITIVE_REQUEST = re.compile(
    rf"^(?:please[,\s]+)?(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
    rf"i\s+(?:want|need|would like)\s+you\s+to\s+|let(?:'s| us)\s+)?{_ACTION}\b"
    r"|^(?:请\s*)?(?:起草|提议|创建|提交|开|标记|给.{1,45}(?:起草|开|建))"
)
_NEGATIVE_REQUEST = re.compile(
    rf"\b(?:do not|don't|never|without|avoid|skip|should not|shouldn't|"
    rf"must not|mustn't|mustn’t)\s+(?:[a-z]+\s+){{0,5}}{_ACTION}\w*\b"
    r"|\b(?:no|without|skip)\s+(?:any\s+)?(?:tickets?|work orders?|proposals?)\b"
    r"|(?:不要|无需|不用|别|不必|请勿).{0,10}(?:工单|提案|提议|更换)"
)
_CONDITIONAL = re.compile(r"\b(?:if|unless|provided that|as long as|only when|assuming)\b|只有|除非|如果")
_SIMPLE_DISK_CONDITION = re.compile(r"\b(?:free|available)\b|剩余|可用")
_UNMODELED_FILTER = re.compile(
    r"\b(?:assigned to|owned by|responsible for|belonging to|employee|department|organizational unit|"
    r"except(?: for)?|excluding|other than)\b|负责人员|归属部门|排除"
)
_ONLY_PLATFORMS = re.compile(
    r"(?:devices?|laptops?|desktops?)\s+(?:running|on|with|using)\s+"
    r"(?:macos|windows|linux)(?:\s+(?:version\s+)?\d+(?:\.\d+){0,3})?\s*[.!?。]?$"
)
_LAST_TARGET = re.compile(
    r"\b(?:the\s+)?last\s+(?:listed\s+)?(?:one|device|laptop|machine|item|entry|result)\b"
    r"|\bthe last\s*[.!?。]?$|最后一台"
)
_ONLY_RESULT = re.compile(
    r"(?:the\s+)?(?:first|second|third|fourth|fifth|last)\s+(?:listed\s+)?"
    r"(?:one|device|laptop|machine|item|entry|result)(?:\s+listed|\s+in the list)?\s*[.!?。]?$"
)
_ONLY_DEVICE = re.compile(r"""(?:the\s+)?(?:device|laptop|machine)\s+(?:id\s*[:=]?\s*)?["']?[\w-]+["']?\s*[.!?。]?$""")
_SIMPLE_RAM_CONDITION = re.compile(r"\b(?:used|usage|utilization|utilisation)\b|已用|使用率|占用")
_READ_REQUEST = re.compile(
    r"^(?:please[,\s]+)?(?:(?:can|could|would|will)\s+you\s+)?"
    r"(?:list|show|find|identify|check|inspect|analy[sz]e|look up|which|how many)\b|^(?:请|帮我)?(?:查|找|列出)"
)
_ACTION_JOIN = re.compile(r"\b(?:and then|and|then)\s+|(?:并且|并|然后)")
_DEFAULT_THRESHOLD = re.compile(r"\b(?:threshold|cutoff|cut-off)\b|阈值")
_ACTION_WORDS = frozenset("""
    a about above affected against all also an and any are at available battery be below both can
    check checks compare compliance consistently constraint could create critical current currently date
    day days desktop desktops device devices disk draft earlier encryption evidence examine fail
    failed failing failure failures fifth file find first flag flagged for fourth free from generate
    give have has health help high historical history how i id identify if in inspect investigate is it its
    last latest laptop laptops least less listed lock look low lower machine machines macos make many
    medium memory more most need now of on one only open or order orders other out over past percent
    persist persistent persistently please prepare pressure proposal propose put raise ram recent
    recently recurring remaining remediation replacement report reported request review running screen
    screen_lock os_up_to_date disk_encryption second service severe severity should show space storage
    sustained system take than that the their them then these they third this those through ticket
    tickets time to today tool trend trends two under up used using usage utilization version volume
    week were what which windows linux will with work would you
    available checks fail os above below exactly exact currently equal previously per threshold cutoff
    """.split())
_MEASUREMENT_UNIT = re.compile(r"\b(free|available|used|occupied|utilization|utilisation|usage)\b|剩余|可用|已用|占用|使用率")
_CONDITION_WORDS = frozenset({
    "a", "all", "and", "are", "available", "be", "capacity", "current", "currently", "device",
    "devices", "disk", "free", "has", "have", "is", "it", "memory", "of", "percent", "pressure",
    "ram", "remaining", "space", "storage", "system", "that", "the", "their", "these", "they", "this",
    "usage", "used", "utilisation", "utilization", "volume", "with",
})


@dataclass(frozen=True)
class Percentage:
    value: float
    comparison: Comparison | None
    conflicting: bool


def explicit_threshold_default(text: str) -> bool:
    return bool(_DEFAULT_THRESHOLD.search(_normalize(text)))


def _normalize(text: str):
    return unicodedata.normalize("NFKC", text).casefold().replace("’", "'")


@dataclass(frozen=True)
class OSConstraint:
    platform: Platform
    exact: str | None
    before: str | None


class UnsupportedScope(ValueError):
    """The request's scope cannot be represented without widening it."""


def requested_os(text: str) -> OSConstraint | None:
    normalized = _normalize(text)
    matches = list(_OS_NAME.finditer(normalized))
    if not matches:
        return None
    if len(matches) != 1:
        if len({(match["family"], match["version"]) for match in matches}) != 1:
            raise UnsupportedScope("Use one OS platform and version per request.")
    match = matches[-1]
    preceding = normalized[:match.start()]
    version = match["version"]
    relative = _OS_RELATIVE_AFTER.match(normalized[match.end():]) if version is None else None
    if relative:
        if relative["direction"] in {"older than", "earlier than", "before", "below", "<", "less than"}:
            return OSConstraint(_OS_PLATFORM[match["family"]], None, relative["version"])
        if relative["direction"] == "equal to":
            return OSConstraint(_OS_PLATFORM[match["family"]], relative["version"], None)
        raise UnsupportedScope("Use an exact OS version or a single earlier-than comparison.")
    if _OS_UNSUPPORTED_PREFIX.search(preceding) or (version and _OS_UNSUPPORTED_SUFFIX.match(normalized[match.end():])):
        raise UnsupportedScope("Use an exact OS version or a single earlier-than comparison.")
    before = bool(version and _OS_BEFORE.search(preceding))
    return OSConstraint(_OS_PLATFORM[match["family"]], None if before else version, version if before else None)


def has_calendar_anchor(text: str) -> bool:
    normalized = _normalize(text)
    return bool(re.search(
        r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|yesterday|"
        r"(?:in|during|since|before|after|as of)\s+(?:19|20)\d{2}|"
        r"(?:as of|at|before|after)\s+[0-2]?\d:[0-5]\d|"
        r"(?:last|this|on|before|after)\s+(?:mon|tues|wednes|thurs|fri|satur|sun)day)\b|昨天|前天",
        normalized,
    ))


def references_last_result(text: str) -> bool:
    return bool(_LAST_TARGET.search(_normalize(text)))


def company_references(text: str) -> list[str]:
    normalized = _normalize(text)
    aliases = re.findall(r"(?<![\w-])(?:acme|globex|initech)(?:-\d{3})?(?![\w-])", normalized)
    qualified = re.findall(r"\b(?:company|tenant)\s+(?:named\s+)?([a-z][\w-]+[-]\d+)\b", normalized)
    return list(dict.fromkeys(aliases + qualified))


def requested_devices(text: str, visible_ids: list[str]) -> list[str]:
    found = []
    for device in visible_ids:
        match = re.search(r"(?<![\w-])" + re.escape(device) + r"(?![\w-])", text, flags=re.IGNORECASE)
        if match:
            found.append((match.start(), device))
    for match in _LABELLED_DEVICE.finditer(text):
        group = match.lastgroup
        device = match[group].rstrip(".")
        if device.casefold() in _TARGET_STOP:
            continue
        # Mark quoted IDs, explicit device-ID phrases and ID-shaped values as targets.
        named = (group != "bare" or bool(re.search(r"[-.\d]", device))
                 or re.search(r"\b(?:for|on|against)\s+(?:the\s+)?$", text[:match.start()], re.IGNORECASE)
                 or re.search(r"\bid\b", text[match.start():match.start(group)], re.IGNORECASE))
        if named:
            found.append((match.start(group), device))
    unique = {}
    for _, device in sorted(found):
        unique.setdefault(device.casefold(), device)
    return list(unique.values())


def percentage_terms(text: str) -> list[Percentage]:
    normalized = _normalize(text)
    terms = []
    for match in _PERCENT.finditer(normalized):
        before = normalized[:match.start()]
        after = normalized[match.end():]
        prefix = next((name for name, pattern in _PREFIX.items()
                       if re.search(rf"(?<![a-z])(?:{pattern})\s*$", before)), None)
        suffix = next((name for name, pattern in _SUFFIX.items()
                       if re.match(rf"\s*(?:{pattern})", after)), None)
        # Competing directions are ambiguous and must not silently fall back.
        conflicting = bool(prefix and suffix and prefix != suffix)
        comparison = None if conflicting else (prefix or suffix)
        terms.append(Percentage(float(match.group(1)), comparison, conflicting))
    return terms


def incompatible_percentage_unit(text: str, metric: str) -> bool:
    if metric not in ("disk", "ram"):
        return False
    normalized = _normalize(text)
    if re.search(r"\d+(?:\.\d+)?\s*(?:[kmgt]i?b|bytes?|megabytes?|gigabytes?|terabytes?)\b|"
                 r"\d+(?:\.\d+)?\s*(?:兆字节|吉字节|千兆字节)", normalized):
        return True
    for match in _PERCENT.finditer(normalized):
        before = list(_MEASUREMENT_UNIT.finditer(normalized[max(0, match.start() - 55):match.start()]))
        after = _MEASUREMENT_UNIT.search(normalized, match.end(), match.end() + 17)
        # A unit immediately following the percentage takes priority ("5% free").
        unit = after.group() if after else (before[-1].group() if before else None)
        if unit is None:
            continue
        if metric == "disk" and unit in {"used", "occupied", "utilization", "utilisation", "usage", "已用", "占用", "使用率"}:
            return True
        if metric == "ram" and unit in {"free", "available", "剩余", "可用"}:
            return True
    return False


def action_block_reason(text: str, action: Action, metric: str, identifiers: list[str]) -> str | None:
    normalized = _normalize(text).strip()
    if _NEGATIVE_REQUEST.search(normalized):
        return "negated_request"
    target = r"\b(?:tickets?|work orders?)\b|工单" if action == "ticket" else r"\breplacement\b|更换"
    first = re.split(r"(?<!\d)[.!?;。！？；]", normalized, maxsplit=1)[0]
    clauses = [first]
    if _READ_REQUEST.match(first):
        clauses.extend(first[match.end():] for match in _ACTION_JOIN.finditer(first)
                       if first[:match.start()].count('"') % 2 == 0)
    if not any(_POSITIVE_REQUEST.match(clause) and re.search(target, clause) for clause in clauses):
        return "no_explicit_action_request"
    if _UNMODELED_FILTER.search(normalized):
        return "unsupported_action_condition"
    if metric == "compliance" and re.search(r"\b(?:was|were|had been)\s+failing\b", normalized):
        return "unsupported_action_condition"
    # Verify the request uses the vocabulary of the implemented action scope.
    # Unknown locations, attributes, and other selectors cannot become fleet-wide drafts.
    action_text = normalized
    for identifier in identifiers:
        action_text = re.sub(r"(?<![\w-])" + re.escape(identifier.casefold()) + r"(?![\w-])", "", action_text)
    unparsed = _PERCENT.sub("", action_text)
    for match in reversed(list(_OS_NAME.finditer(unparsed))):
        relative = _OS_RELATIVE_AFTER.match(unparsed[match.end():]) if not match["version"] else None
        end = match.end() + relative.end() if relative else match.end()
        unparsed = unparsed[:match.start()] + match["family"] + unparsed[end:]
    unparsed = re.sub(r"\b(?:[2-9]|[12]\d|30)[-\s]+days?\b", "", unparsed)
    words = set(re.findall(r"[a-z_]+", action_text))
    if words - _ACTION_WORDS or re.search(r"\d|[\u3400-\u9fff]", unparsed):
        return "unsupported_action_condition"
    scope = re.search(r"\bonly (?:on|for|against)\s+(?P<target>.+)$", normalized)
    if scope and not any(
        matcher.fullmatch(scope["target"]) for matcher in (_ONLY_PLATFORMS, _ONLY_DEVICE, _ONLY_RESULT)
    ):
        return "unsupported_action_condition"
    marker = _CONDITIONAL.search(normalized)
    if marker is None:
        return None
    if marker.group() not in {"if", "only when", "如果", "只有"} or metric not in {"disk", "ram"}:
        return "unsupported_action_condition"
    condition = normalized[marker.end():].strip()
    terms = percentage_terms(condition)
    units = _SIMPLE_DISK_CONDITION if metric == "disk" else _SIMPLE_RAM_CONDITION
    if len(terms) != 1 or terms[0].comparison is None or not units.search(condition):
        return "unsupported_action_condition"
    condition = re.sub(r"\b(?:or|and)\s+(?:more|above|higher|greater|less|below|lower|fewer|under)\b",
                       "", condition)
    for pattern in _PREFIX.values():
        condition = re.sub(rf"(?<![a-z])(?:{pattern})(?![a-z])", "", condition)
    condition = _PERCENT.sub("", condition)
    words = set(re.findall(r"\b[a-z]+\b", condition))
    if words - _CONDITION_WORDS or "and" in words:
        return "unsupported_action_condition"
    if re.search(r"[^\x00-\x7f]", condition):
        return "unsupported_action_condition"
    return None
