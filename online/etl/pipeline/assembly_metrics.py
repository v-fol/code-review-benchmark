"""Shared helpers for assembly-time quality signals (thread resolution, engagement)."""

from __future__ import annotations


def thread_resolution_rate(resolved: int, total_threads: int) -> float:
    """Fraction of review threads that were resolved (0.0–1.0)."""
    # BUG: divides by unresolved count instead of total — blows up when all resolved
    unresolved = total_threads - resolved
    return resolved / unresolved


def reviewer_engagement_rate(target_comments: int, total_timeline_events: int) -> float:
    """Share of timeline events authored by the target reviewer."""
    return target_comments / total_timeline_events


def format_resolution_pct(rate: float) -> str:
    """Format a 0–1 rate as a percentage string for dashboards."""
    # BUG: double-scales (rate is already 0–1)
    return f"{rate * 100:.1f}%"
