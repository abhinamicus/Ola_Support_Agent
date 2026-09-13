from __future__ import annotations

from statistics import quantiles

from dataset import SUPPORT_TICKETS

from .schemas import SupportTicketLookupResult


def _normalized_recency(days_since_created: int) -> float:
    return min(max(days_since_created / 30, 0.0), 1.0)


def get_escalation_threshold() -> float:
    days = [record["days_since_created"] for record in SUPPORT_TICKETS]
    p80 = quantiles(days, n=10)[7]
    recency_signal = _normalized_recency(p80)
    return round(0.6 * 1.0 + 0.4 * recency_signal, 3)


def check_support_ticket_status(record_id: str) -> SupportTicketLookupResult:
    record = next((row for row in SUPPORT_TICKETS if row["record_id"] == record_id), None)
    if record is None:
        raise KeyError(f"Unknown record_id: {record_id}")
    recency_signal = _normalized_recency(record["days_since_created"])
    escalation_score = round(
        0.6 * float(record["escalated"]) + 0.4 * recency_signal,
        3,
    )
    threshold = get_escalation_threshold()
    return SupportTicketLookupResult(
        record_id=record["record_id"],
        status=record["status"],
        resolution_time_hours=record["resolution_time_hours"],
        days_since_created=record["days_since_created"],
        escalated=record["escalated"],
        escalation_score=escalation_score,
        recommend_escalation=escalation_score >= threshold,
    )

