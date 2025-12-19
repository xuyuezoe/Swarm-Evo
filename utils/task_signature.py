"""Task signature inference helpers."""

from __future__ import annotations

import re
from typing import Dict, List


def infer_signature_from_description(text: str) -> Dict[str, str]:
    """Infer a coarse task signature from the problem description."""
    description = (text or "").lower()

    input_type = "unknown"
    if "image" in description or "pixel" in description:
        input_type = "image"
    elif "tabular" in description or "csv" in description or "table" in description:
        input_type = "tabular"

    label_type = "unknown"
    if "binary" in description or "two classes" in description:
        label_type = "binary"
    elif "multiclass" in description or "multi-class" in description:
        label_type = "multiclass"

    resolution = ""
    match = re.search(r"(\d{2,4})\s*[xX]\s*(\d{2,4})", description)
    if match:
        resolution = f"{match.group(1)}x{match.group(2)}"

    domain_keywords = _extract_domain_keywords(description)
    domain = ",".join(domain_keywords) if domain_keywords else "unknown"

    return {
        "input_type": input_type,
        "label_type": label_type,
        "resolution": resolution,
        "domain": domain,
    }


def signature_to_text(signature: Dict[str, str]) -> str:
    """Render a signature dict into a stable text representation."""
    items = []
    for key in sorted(signature.keys()):
        value = signature[key]
        items.append(f"{key}={value}")
    return "; ".join(items)


def _extract_domain_keywords(description: str) -> List[str]:
    tokens = [
        token.strip()
        for token in re.split(r"[^a-z0-9]+", description)
        if token.strip()
    ]
    keywords = []
    for token in tokens:
        if token in {"image", "tabular", "classification", "binary", "multiclass"}:
            continue
        if token in keywords:
            continue
        keywords.append(token)
        if len(keywords) >= 2:
            break
    return keywords
