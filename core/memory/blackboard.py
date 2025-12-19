"""Global blackboard persistence and task retrieval logic."""

from __future__ import annotations

import json
import math
import os
import threading
from typing import Dict, List, Sequence, Tuple

from .pheromone import decay_stats, pheromone_strength, update_stats
from .types import GeneStats, NodeSummary


class GlobalBlackboard:
    """Persistent memory for storing task outcomes and gene pheromones."""

    VERSION = "v1.1"
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        path: str,
        topk_similar_tasks: int = 5,
        embedding_model_name: str | None = None,
    ) -> None:
        self.path = path
        self.topk_similar_tasks = topk_similar_tasks
        self.embedding_model_name = embedding_model_name or self.DEFAULT_EMBEDDING_MODEL

        self.version = self.VERSION
        self.tasks: Dict[str, Dict] = {}
        self.genes: Dict[str, Dict[str, GeneStats]] = {}

        self._embedding_model = None
        self._embedding_failed = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: str,
        topk_similar_tasks: int = 5,
        embedding_model_name: str | None = None,
    ) -> "GlobalBlackboard":
        board = cls(path, topk_similar_tasks, embedding_model_name)
        if not os.path.exists(path):
            return board

        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        board.version = data.get("version", board.VERSION)
        board.tasks = data.get("tasks", {}) or {}
        raw_genes = data.get("genes", {}) or {}

        for gene_key, value_map in raw_genes.items():
            board.genes[gene_key] = {}
            if not isinstance(value_map, dict):
                continue
            for gene_value, stats_dict in value_map.items():
                board.genes[gene_key][gene_value] = board._dict_to_stats(stats_dict)

        return board

    def save(self) -> None:
        """Persist the blackboard state to disk."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        gene_payload: Dict[str, Dict[str, Dict[str, float]]] = {}
        for gene_key, value_map in self.genes.items():
            gene_payload[gene_key] = {}
            for gene_value, stats in value_map.items():
                gene_payload[gene_key][gene_value] = self._stats_to_dict(stats)

        payload = {
            "version": self.version,
            "tasks": self.tasks,
            "genes": gene_payload,
        }

        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Task registration and updates
    # ------------------------------------------------------------------
    def register_task(
        self,
        task_id: str,
        signature_json: Dict,
        signature_text: str,
        created_at: float,
    ) -> None:
        """Ensure a task entry exists and has an embedding."""
        with self._lock:
            record = self.tasks.get(task_id)
            if record is None:
                record = {
                    "task_id": task_id,
                    "signature_json": signature_json,
                    "signature_text": signature_text,
                    "created_at": created_at,
                    "positives": [],
                    "negatives": [],
                    "embedding": self.embed_signature(signature_text),
                }
                self.tasks[task_id] = record
            else:
                record["signature_json"] = signature_json
                record["signature_text"] = signature_text
                record.setdefault("positives", [])
                record.setdefault("negatives", [])
                if not record.get("embedding"):
                    record["embedding"] = self.embed_signature(signature_text)

    def update_from_summaries(
        self,
        task_id: str,
        signature_json: Dict,
        signature_text: str,
        positives: Sequence[NodeSummary],
        negatives: Sequence[NodeSummary],
        current_step: int,
    ) -> None:
        """Update the persistent task record and global gene stats."""
        self.register_task(task_id, signature_json, signature_text, created_at=0.0)

        with self._lock:
            record = self.tasks[task_id]
            record["positives"] = [self._summary_to_dict(summary) for summary in positives]
            record["negatives"] = [self._summary_to_dict(summary) for summary in negatives]

            for summary in list(positives) + list(negatives):
                for gene_key, gene_value in summary.gene_values.items():
                    gene_bucket = self.genes.setdefault(gene_key, {})
                    stats = gene_bucket.get(gene_value)
                    if stats is None:
                        stats = GeneStats()
                    stats = decay_stats(stats, current_step, decay=0.8)
                    stats = update_stats(stats, summary.score, summary.is_buggy, current_step)
                    gene_bucket[gene_value] = stats

    # ------------------------------------------------------------------
    # Similar task retrieval & aggregation
    # ------------------------------------------------------------------
    def embed_signature(self, text: str) -> List[float]:
        """Return an embedding for the provided signature text if models are available."""
        text = (text or "").strip()
        if not text:
            return []
        model = self._get_embedding_model()
        if model is None:
            return []
        try:
            vector = model.encode(text, normalize_embeddings=True)
        except Exception:
            self._embedding_failed = True
            self._embedding_model = None
            return []
        return [float(v) for v in vector]

    def retrieve_similar_tasks(
        self, signature_text: str, topk: int | None = None
    ) -> List[Tuple[str, float]]:
        """Return up to top-k tasks most similar to the provided signature."""
        if not self.tasks:
            return []

        topk = topk or self.topk_similar_tasks
        query_embedding = self.embed_signature(signature_text)
        use_embeddings = bool(query_embedding)
        similarities: List[Tuple[str, float]] = []

        for task_id, record in self.tasks.items():
            record_text = record.get("signature_text", "")
            if use_embeddings:
                record_embedding = record.get("embedding")
                if not record_embedding and record_text:
                    record_embedding = self.embed_signature(record_text)
                    record["embedding"] = record_embedding
                if not record_embedding:
                    continue
                if len(record_embedding) != len(query_embedding):
                    continue
                sim = self._cosine_similarity(query_embedding, record_embedding)
            else:
                sim = self._jaccard_similarity(signature_text, record_text)
            if sim <= 0.0:
                continue
            similarities.append((task_id, sim))

        similarities.sort(key=lambda item: item[1], reverse=True)
        return similarities[:topk]

    def aggregate_gene_distribution(
        self, similar_tasks: Sequence[Tuple[str, float]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Aggregate pheromone strengths across the provided similar task list."""
        aggregated: Dict[str, Dict[str, float]] = {}

        for task_id, similarity in similar_tasks:
            record = self.tasks.get(task_id)
            if not record:
                continue
            summaries = list(record.get("positives", [])) + list(record.get("negatives", []))
            for summary_dict in summaries:
                gene_values = summary_dict.get("gene_values", {}) or {}
                for gene_key, gene_value in gene_values.items():
                    stats = self.genes.get(gene_key, {}).get(gene_value)
                    if not stats:
                        continue
                    strength = pheromone_strength(stats)
                    bucket = aggregated.setdefault(gene_key, {})
                    bucket[gene_value] = bucket.get(gene_value, 0.0) + similarity * strength

        ranked: Dict[str, List[Tuple[str, float]]] = {}
        for gene_key, value_map in aggregated.items():
            ordered = sorted(
                value_map.items(), key=lambda item: (item[1], item[0]), reverse=True
            )
            ranked[gene_key] = ordered[:2]
        return ranked

    def summarize_for_prompt(self, signature_text: str) -> str:
        """Create a deterministic global prior string for task prompts."""
        similar = self.retrieve_similar_tasks(signature_text, topk=self.topk_similar_tasks)
        if not similar:
            return ""

        aggregated = self.aggregate_gene_distribution(similar)
        if not aggregated:
            return ""

        # Rank gene keys by the strongest preference value
        ranked_keys = sorted(
            [
                (gene_key, values[0][1] if values else 0.0)
                for gene_key, values in aggregated.items()
            ],
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )

        # Limit to at most 5 bullets
        bullet_keys = [gene_key for gene_key, _ in ranked_keys[:5]]
        sections: List[str] = []
        for gene_key in bullet_keys:
            values = aggregated.get(gene_key, [])
            if not values:
                continue
            prefer_value = values[0][0]
            avoid_value = values[1][0] if len(values) > 1 else "unexplored variants"
            sections.append(f"- {gene_key}: prefer {prefer_value}; avoid {avoid_value}")

        if not sections:
            return ""

        header = f"Global Prior (from {len(similar)} similar tasks):"
        return "\n".join([header] + sections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _stats_to_dict(self, stats: GeneStats) -> Dict[str, float]:
        return {
            "usage": stats.usage,
            "success": stats.success,
            "fail": stats.fail,
            "ema_score": stats.ema_score,
            "last_step": stats.last_step,
        }

    def _dict_to_stats(self, payload: Dict) -> GeneStats:
        return GeneStats(
            usage=float(payload.get("usage", 0.0) or 0.0),
            success=float(payload.get("success", 0.0) or 0.0),
            fail=float(payload.get("fail", 0.0) or 0.0),
            ema_score=float(payload.get("ema_score", 0.0) or 0.0),
            last_step=int(payload.get("last_step", 0) or 0),
        )

    def _summary_to_dict(self, summary: NodeSummary) -> Dict:
        return {
            "node_id": summary.node_id,
            "gene_values": dict(summary.gene_values),
            "score": summary.score,
            "is_buggy": summary.is_buggy,
            "summary": summary.summary,
            "step": summary.step,
            "signature_hash": summary.signature_hash,
        }

    def _get_embedding_model(self):
        if self._embedding_failed:
            return None
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            self._embedding_failed = True
            return None
        try:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        except Exception:
            self._embedding_failed = True
            self._embedding_model = None
            return None
        return self._embedding_model

    def _cosine_similarity(
        self, vector_a: Sequence[float], vector_b: Sequence[float]
    ) -> float:
        numerator = sum(x * y for x, y in zip(vector_a, vector_b))
        denom_a = math.sqrt(sum(x * x for x in vector_a))
        denom_b = math.sqrt(sum(x * x for x in vector_b))
        if denom_a <= 0.0 or denom_b <= 0.0:
            return 0.0
        return numerator / (denom_a * denom_b)

    def _jaccard_similarity(self, text_a: str, text_b: str) -> float:
        set_a = self._tokenize(text_a)
        set_b = self._tokenize(text_b)
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return intersection / union

    def _tokenize(self, text: str) -> set[str]:
        tokens = []
        current = []
        for ch in (text or "").lower():
            if ch.isalnum():
                current.append(ch)
            elif current:
                tokens.append("".join(current))
                current = []
        if current:
            tokens.append("".join(current))
        return set(tokens)
