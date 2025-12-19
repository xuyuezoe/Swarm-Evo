"""In-memory cache for deterministic pheromone aggregation."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from core.execution.journal import Node

from .types import NodeSummary


@dataclass(frozen=True)
class _SummaryRecord:
    summary: NodeSummary
    strength: float
    quality: float
    stability: float
    novelty: float


class ExperienceCache:
    """Thread-safe cache that stores the best reviewed nodes per gene signature."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._summaries: Dict[str, NodeSummary] = {}
        self._usage_counts: Dict[str, int] = {}

    def add_review(self, node: Node, gene_values: Dict[str, str]) -> None:
        """Add a reviewed node summary, deduplicating by gene signature."""
        signature_hash = self._hash_gene_values(gene_values)
        summary_text = (node.summary or "").strip()
        summary = NodeSummary(
            node_id=node.id,
            gene_values=dict(gene_values),
            score=node.score,
            is_buggy=bool(node.is_buggy),
            summary=summary_text,
            step=node.step,
            signature_hash=signature_hash,
        )

        with self._lock:
            self._usage_counts[signature_hash] = self._usage_counts.get(signature_hash, 0) + 1
            existing = self._summaries.get(signature_hash)
            if existing is None or self._is_better(summary, existing):
                self._summaries[signature_hash] = summary

    def select_top_pos_neg(self) -> Tuple[List[NodeSummary], List[NodeSummary]]:
        """Return the positive/negative selections without clearing the cache."""
        with self._lock:
            summaries = list(self._summaries.values())
            usage_counts = dict(self._usage_counts)
        return self._rank_summaries(summaries, usage_counts)

    def flush(self) -> Tuple[List[NodeSummary], List[NodeSummary]]:
        """Return selections and reset the cache."""
        with self._lock:
            summaries = list(self._summaries.values())
            usage_counts = dict(self._usage_counts)
            self._summaries.clear()
            self._usage_counts.clear()
        return self._rank_summaries(summaries, usage_counts)

    def _rank_summaries(
        self, summaries: Iterable[NodeSummary], usage_counts: Dict[str, int]
    ) -> Tuple[List[NodeSummary], List[NodeSummary]]:
        records = [self._build_record(summary, usage_counts) for summary in summaries]
        positives = self._select_positives(records)
        negatives = self._select_negatives(records)
        return positives, negatives

    def _select_positives(self, records: List[_SummaryRecord]) -> List[NodeSummary]:
        selected: List[NodeSummary] = []
        used_ids: set[str] = set()

        positive_records = [
            record
            for record in records
            if not record.summary.is_buggy and record.summary.score is not None
        ]
        positive_records.sort(
            key=lambda r: (
                r.strength,
                r.quality,
                r.novelty,
                r.summary.step,
                r.summary.node_id,
            ),
            reverse=True,
        )
        for record in positive_records[:5]:
            selected.append(record.summary)
            used_ids.add(record.summary.node_id)

        if len(selected) < 5:
            fallback_records = [
                record
                for record in records
                if not record.summary.is_buggy and record.summary.node_id not in used_ids
            ]
            fallback_records.sort(
                key=lambda r: (
                    r.stability,
                    r.novelty,
                    r.summary.step,
                    r.summary.node_id,
                ),
                reverse=True,
            )
            for record in fallback_records:
                if len(selected) >= 5:
                    break
                selected.append(record.summary)
                used_ids.add(record.summary.node_id)

        if len(selected) < 5:
            buggy_records = [
                record
                for record in records
                if record.summary.is_buggy and record.summary.node_id not in used_ids
            ]
            buggy_records.sort(
                key=lambda r: (r.novelty, r.summary.step, r.summary.node_id), reverse=True
            )
            for record in buggy_records:
                if len(selected) >= 5:
                    break
                selected.append(record.summary)
                used_ids.add(record.summary.node_id)

        return selected

    def _select_negatives(self, records: List[_SummaryRecord]) -> List[NodeSummary]:
        selected: List[NodeSummary] = []
        used_ids: set[str] = set()

        negative_records = [
            record
            for record in records
            if record.summary.is_buggy or record.summary.score is None
        ]
        negative_records.sort(
            key=lambda r: (
                r.strength,
                r.quality,
                r.novelty,
                r.summary.step,
                r.summary.node_id,
            ),
            reverse=True,
        )
        for record in negative_records[:2]:
            selected.append(record.summary)
            used_ids.add(record.summary.node_id)

        if len(selected) < 2:
            fallback_records = [
                record for record in records if record.summary.node_id not in used_ids
            ]
            fallback_records.sort(
                key=lambda r: (
                    0 if r.summary.is_buggy else 1,
                    r.quality,
                    r.strength,
                    r.summary.step,
                    r.summary.node_id,
                )
            )
            for record in fallback_records:
                if len(selected) >= 2:
                    break
                selected.append(record.summary)
                used_ids.add(record.summary.node_id)

        return selected

    def _build_record(
        self, summary: NodeSummary, usage_counts: Dict[str, int]
    ) -> _SummaryRecord:
        usage = max(usage_counts.get(summary.signature_hash, 0), 0)
        novelty = 1.0 / math.sqrt(1.0 + usage)
        quality = self._quality(summary)
        stability = 0.0 if summary.is_buggy else 1.0
        strength = 0.7 * quality + 0.2 * stability + 0.1 * novelty
        return _SummaryRecord(summary, strength, quality, stability, novelty)

    def _quality(self, summary: NodeSummary) -> float:
        if summary.score is None:
            return 0.0
        return max(0.0, min(1.0, summary.score))

    def _is_better(self, candidate: NodeSummary, incumbent: NodeSummary) -> bool:
        candidate_key = (
            0 if candidate.is_buggy else 1,
            self._quality(candidate),
            candidate.step,
            candidate.node_id,
        )
        incumbent_key = (
            0 if incumbent.is_buggy else 1,
            self._quality(incumbent),
            incumbent.step,
            incumbent.node_id,
        )
        return candidate_key > incumbent_key

    def _hash_gene_values(self, gene_values: Dict[str, str]) -> str:
        serialized = json.dumps(sorted(gene_values.items()), separators=(",", ":"))
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()
