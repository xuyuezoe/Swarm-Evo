"""Shared dataclasses for the pheromone memory system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GeneStats:
    """Aggregated pheromone statistics for a single gene."""

    usage: float = 0.0
    success: float = 0.0
    fail: float = 0.0
    ema_score: float = 0.0
    last_step: int = 0


@dataclass
class NodeSummary:
    """Compact, non-code summary of a reviewed node."""

    node_id: str
    gene_values: Dict[str, str]
    score: Optional[float]
    is_buggy: bool
    summary: str
    step: int
    signature_hash: str


@dataclass
class TaskRecord:
    """Persistent representation of task-level memory."""

    task_id: str
    signature_json: Dict
    signature_text: str
    created_at: float
    positives: List[NodeSummary] = field(default_factory=list)
    negatives: List[NodeSummary] = field(default_factory=list)
