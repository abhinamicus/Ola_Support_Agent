from __future__ import annotations

import json
import random
from collections import Counter
from statistics import quantiles

SEED = 20260829
CATEGORY_WEIGHTS = {
    "Billing": 0.24,
    "Technical Issue": 0.22,
    "Account Access": 0.18,
    "Product Defect": 0.16,
    "General Inquiry": 0.20,
}
STATUS_WEIGHTS = {
    "Open": 0.22,
    "In Progress": 0.24,
    "Escalated": 0.12,
    "Resolved": 0.24,
    "Closed": 0.18,
}
RESOLUTION_TIME_RANGE_HOURS = (2.0, 144.0)
RECORD_COUNT = 48
CATEGORY_STATUS_PAIRS = [
    ("Billing", "Open"),
    ("Technical Issue", "In Progress"),
    ("Account Access", "Resolved"),
    ("Product Defect", "Escalated"),
    ("General Inquiry", "Closed"),
]


def _weighted_choice(rng: random.Random, weighted_values: dict[str, float]) -> str:
    values = list(weighted_values.keys())
    weights = list(weighted_values.values())
    return rng.choices(values, weights=weights, k=1)[0]


def generate_support_tickets(
    seed: int = SEED,
    record_count: int = RECORD_COUNT,
) -> list[dict]:
    rng = random.Random(seed)
    tickets: list[dict] = []

    for index, (category, status) in enumerate(CATEGORY_STATUS_PAIRS, start=1):
        tickets.append(
            {
                "record_id": f"OLA-{index:04d}",
                "category": category,
                "status": status,
                "resolution_time_hours": round(rng.uniform(*RESOLUTION_TIME_RANGE_HOURS), 1),
                "days_since_created": rng.randint(0, 30),
                "escalated": status == "Escalated" or rng.random() < 0.18,
            }
        )

    for index in range(len(tickets) + 1, record_count + 1):
        status = _weighted_choice(rng, STATUS_WEIGHTS)
        escalated = status == "Escalated" or rng.random() < 0.14
        if escalated and status in {"Resolved", "Closed"} and rng.random() < 0.55:
            status = "Escalated"
        tickets.append(
            {
                "record_id": f"OLA-{index:04d}",
                "category": _weighted_choice(rng, CATEGORY_WEIGHTS),
                "status": status,
                "resolution_time_hours": round(rng.uniform(*RESOLUTION_TIME_RANGE_HOURS), 1),
                "days_since_created": rng.randint(0, 30),
                "escalated": escalated,
            }
        )

    return tickets


SUPPORT_TICKETS = generate_support_tickets()


def compute_stats(records: list[dict]) -> dict:
    category_counts = Counter(record["category"] for record in records)
    status_counts = Counter(record["status"] for record in records)
    escalated_pct = round(
        100 * sum(1 for record in records if record["escalated"]) / len(records), 2
    )
    days = [record["days_since_created"] for record in records]
    p80_days = quantiles(days, n=10)[7]
    return {
        "seed": SEED,
        "record_count": len(records),
        "category_counts": dict(category_counts),
        "status_counts": dict(status_counts),
        "escalated_percentage": escalated_pct,
        "days_since_created_p80": p80_days,
        "resolution_time_range_hours": RESOLUTION_TIME_RANGE_HOURS,
    }


def validate_dataset(records: list[dict]) -> None:
    category_counts = Counter(record["category"] for record in records)
    status_counts = Counter(record["status"] for record in records)
    assert len(records) >= 40
    for category in CATEGORY_WEIGHTS:
        assert category_counts[category] >= 3, f"category threshold failed for {category}"
    for status in STATUS_WEIGHTS:
        assert status_counts[status] >= 1, f"status coverage failed for {status}"
    escalated_pct = 100 * sum(1 for record in records if record["escalated"]) / len(records)
    assert 10 <= escalated_pct <= 30, "escalated percentage outside accepted band"


def main() -> None:
    validate_dataset(SUPPORT_TICKETS)
    stats = compute_stats(SUPPORT_TICKETS)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
