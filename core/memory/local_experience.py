"""Local journal ranking and deterministic report generation."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from core.execution.journal import Journal, Node

from .gene_normalizer import GeneNormalizer


def compute_node_strength(node: Node, novelty_within_task: float) -> float:
    """Blend quality, stability, and novelty into a single score."""
    score = node.score if node.score is not None else 0.0
    quality = max(0.0, min(1.0, score))
    stability = 1.0 if not node.is_buggy else 0.0
    novelty = max(0.0, float(novelty_within_task))
    return 0.7 * quality + 0.2 * stability + 0.1 * novelty


def select_top_nodes(
    journal: Journal, gene_normalizer: GeneNormalizer, topn: int = 5
) -> List[Node]:
    """Deterministically select the strongest nodes for local context."""
    nodes = [
        node
        for node in journal.nodes.values()
        if node.code and node.code.strip()
    ]
    if not nodes:
        return []

    signature_cache: Dict[str, Tuple[Dict[str, str], Tuple[Tuple[str, str], ...]]] = {}
    signature_counts: Dict[Tuple[Tuple[str, str], ...], int] = {}

    for node in nodes:
        gene_values = gene_normalizer.discretize(node.genes or {})
        signature = tuple(sorted(gene_values.items()))
        signature_cache[node.id] = (gene_values, signature)
        signature_counts[signature] = signature_counts.get(signature, 0) + 1

    strength_records: List[Tuple[Node, float]] = []
    for node in nodes:
        _, signature = signature_cache[node.id]
        usage = signature_counts.get(signature, 0)
        novelty = 1.0 / math.sqrt(1.0 + float(usage))
        strength = compute_node_strength(node, novelty)
        strength_records.append((node, strength))

    strength_records.sort(
        key=lambda item: (
            item[1],
            0 if item[0].is_buggy else 1,
            0 if item[0].score is None else 1,
            item[0].step,
            item[0].id,
        ),
        reverse=True,
    )
    selected: List[Node] = [node for node, _ in strength_records[:topn]]
    selected_ids = {node.id for node in selected}

    if len(selected) < topn:
        recent_nodes = sorted(
            nodes,
            key=lambda node: (node.step, node.id),
            reverse=True,
        )
        for node in recent_nodes:
            if len(selected) >= topn:
                break
            if node.id in selected_ids:
                continue
            selected.append(node)
            selected_ids.add(node.id)

    return selected


def generate_local_report(nodes: Sequence[Node], gene_normalizer: GeneNormalizer) -> str:
    """Generate a deterministic text report summarizing top journal experience."""
    header = "Local Report (top candidates):"
    if not nodes:
        return "\n".join(
            [
                header,
                "Recommended MODEL: insufficient data (common among top candidates)",
                "Recommended DATA: insufficient data",
                "Action: keep structure tags and ensure validation score printed.",
            ]
        )

    model_counter: Counter[str] = Counter()
    data_counter: Counter[str] = Counter()
    risky_flags: List[str] = []

    for node in nodes:
        gene_values = gene_normalizer.discretize(node.genes or {})
        model_counter[gene_values.get("MODEL", "UnknownModel")] += 1
        data_counter[gene_values.get("DATA", "GenericData")] += 1

        summary_text = (node.summary or "").lower()
        if node.is_buggy:
            risky_flags.append("buggy candidates observed")
        if "filenotfound" in summary_text:
            risky_flags.append("FileNotFound errors reported")
        if "shape mismatch" in summary_text:
            risky_flags.append("shape mismatch mentioned")

    model_label = model_counter.most_common(1)[0][0] if model_counter else "UnknownModel"
    data_label = data_counter.most_common(1)[0][0] if data_counter else "GenericData"

    lines = [
        header,
        f"Recommended MODEL: {model_label} (common among top candidates)",
        f"Recommended DATA: {data_label}",
    ]

    if risky_flags:
        deduped_risks = []
        seen = set()
        for flag in risky_flags:
            if flag in seen:
                continue
            deduped_risks.append(flag)
            seen.add(flag)
        lines.append(f"Risky patterns: {', '.join(deduped_risks)}")

    lines.append("Action: keep structure tags and ensure validation score printed.")
    return "\n".join(lines)
