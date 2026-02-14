import json
import re
from typing import Any, Dict, Iterator, List
import logging

logger = logging.getLogger(__name__)


def _iter_balanced_json_substrings(text: str) -> Iterator[str]:
    """Yield balanced JSON object/array substrings found in text."""
    stack = []
    start = None
    pairs = {"]": "[", "}": "{"}
    for i, ch in enumerate(text):
        if ch in "{[":
            if not stack:
                start = i
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            if stack[-1] != pairs[ch]:
                stack = []
                start = None
                continue
            stack.pop()
            if not stack and start is not None:
                yield text[start:i + 1]
                start = None


def _try_parse_json(candidate: str) -> Any:
    """Attempt to parse JSON with small repairs."""
    try:
        return json.loads(candidate)
    except Exception:
        pass

    repaired = re.sub(r",\s*([\}\]])", r"\1", candidate)
    try:
        return json.loads(repaired)
    except Exception:
        pass


    repaired2 = repaired.replace("'", '"')
    try:
        return json.loads(repaired2)
    except Exception:
        return None


def safe_json_load(text: str) -> Any:
    """Best-effort JSON extraction from noisy model output."""
    if not text:
        return None

    cleaned = text.strip()
    if not cleaned:
        return None

    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE)

    cleaned = re.sub(r"^(?:Assistant|Human|Response|Output|Answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = cleaned.strip()

    parsed = _try_parse_json(cleaned)
    if parsed is not None:
        return parsed

    for candidate in _iter_balanced_json_substrings(cleaned):
        parsed = _try_parse_json(candidate)
        if parsed is not None:
            return parsed

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", cleaned):
        start = match.start()
        candidate = cleaned[start:]
        try:
            obj, _ = decoder.raw_decode(candidate)
            return obj
        except json.JSONDecodeError:
            continue


    logger.debug(f"safe_json_load failed to parse: {cleaned[:200]}...")
    return None


def _ensure_dict(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def _ensure_list(obj: Any) -> List[Any]:
    return obj if isinstance(obj, list) else []


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_json_response(raw: Any) -> Dict[str, Any]:
    """
    Accepts either a parsed object or a raw string and returns a dict.
    If parsing fails, returns an empty dict.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = safe_json_load(raw)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def normalize_utility_tag(raw: Any) -> Dict[str, Any]:
    """
    Normalizes utility tagger output to a stable dict shape.
    """
    data = _ensure_dict(raw)
    entities = _ensure_list(data.get("entities"))
    return {
        "importance": _to_float(data.get("importance", 0.0)),
        "surprise": _to_float(data.get("surprise", 0.0)),
        "emotion": _to_float(data.get("emotion", 0.0)),
        "entities": entities,
    }


def normalize_entity_relation(raw: Any) -> Dict[str, List[Dict[str, str]]]:
    """
    Normalizes entity + relation extraction output to stable lists.
    """
    data = _ensure_dict(raw)
    entities_in = _ensure_list(data.get("entities"))
    relations_in = _ensure_list(data.get("relations"))

    entities: List[Dict[str, str]] = []
    for ent in entities_in:
        if isinstance(ent, dict):
            name = str(ent.get("name", "")).strip()
            etype = str(ent.get("type", "Other")).strip() if ent.get("type") else "Other"
        else:
            name = str(ent).strip()
            etype = "Other"
        if name:
            entities.append({"name": name, "type": etype})

    relations: List[Dict[str, str]] = []
    for rel in relations_in:
        if not isinstance(rel, dict):
            continue
        source = str(rel.get("source", "")).strip()
        target = str(rel.get("target", "")).strip()
        relation = str(rel.get("relation", "")).strip()
        if not source or not target or not relation:
            continue
        relations.append({
            "source": source,
            "target": target,
            "relation": relation,
        })

    return {"entities": entities, "relations": relations}
