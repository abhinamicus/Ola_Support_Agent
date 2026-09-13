from __future__ import annotations

import re
from dataclasses import dataclass

PHONE_PATTERN = re.compile(r"\b\d{10}\b")
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|previous) instructions", re.IGNORECASE),
    re.compile(r"reveal (the )?system prompt", re.IGNORECASE),
    re.compile(r"bypass (your )?(guardrails|rules)", re.IGNORECASE),
]


@dataclass
class GuardrailResult:
    sanitized_text: str
    events: list[str]
    blocked: bool = False
    reason: str | None = None


def mask_phone_numbers(text: str) -> tuple[str, bool]:
    masked = PHONE_PATTERN.sub("******MASKED******", text)
    return masked, masked != text


def apply_input_guardrails(text: str) -> GuardrailResult:
    sanitized_text, pii_masked = mask_phone_numbers(text)
    events: list[str] = []
    if pii_masked:
        events.append("pii_masked_phone_number")

    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(sanitized_text):
            events.append("prompt_injection_detected")
            return GuardrailResult(
                sanitized_text=sanitized_text,
                events=events,
                blocked=True,
                reason="Potential prompt injection attempt detected.",
            )

    return GuardrailResult(sanitized_text=sanitized_text, events=events)

