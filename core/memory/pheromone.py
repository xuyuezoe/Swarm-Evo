"""Pheromone math helpers for gene-level statistics."""

from __future__ import annotations

import math

from .types import GeneStats


def update_stats(
    stats: GeneStats, score: float | None, is_buggy: bool, current_step: int
) -> GeneStats:
    """Update a gene's stats based on the latest observation."""
    stats.usage += 1.0
    if not is_buggy and score is not None:
        stats.success += 1.0
    else:
        stats.fail += 1.0

    if score is not None:
        if stats.ema_score == 0.0:
            stats.ema_score = score
        else:
            stats.ema_score = 0.9 * stats.ema_score + 0.1 * score

    stats.last_step = current_step
    return stats


def decay_stats(stats: GeneStats, current_step: int, decay: float = 0.8) -> GeneStats:
    """Apply global decay when steps advance to keep stats bounded."""
    if current_step > stats.last_step:
        stats.usage *= decay
        stats.success *= decay
        stats.fail *= decay
    stats.last_step = current_step
    return stats


def compute_quality(stats: GeneStats) -> float:
    """Return a normalized quality estimate from the EMA score."""
    score = stats.ema_score
    if score <= 0.0:
        return 0.0
    if 0.0 <= score <= 1.0:
        return max(0.0, min(1.0, score))
    return score


def compute_stability(stats: GeneStats) -> float:
    """Estimate stability as the empirical success ratio."""
    denom = stats.success + stats.fail
    if denom <= 0.0:
        return 0.5
    return stats.success / denom


def compute_novelty(stats: GeneStats) -> float:
    """Penalize heavily reused genes while rewarding new combinations."""
    return 1.0 / math.sqrt(1.0 + max(stats.usage, 0.0))


def pheromone_strength(
    stats: GeneStats, wq: float = 0.6, ws: float = 0.3, wn: float = 0.1
) -> float:
    """Blend the individual components into a single pheromone score."""
    quality = compute_quality(stats)
    stability = compute_stability(stats)
    novelty = compute_novelty(stats)
    return wq * quality + ws * stability + wn * novelty
