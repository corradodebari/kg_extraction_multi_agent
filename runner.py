"""
WayFlow step-based execution runner.
Ties together tools, memory, and database connections.
"""

import json
import hashlib
import os
import re
import sys
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import anyio
import oracledb
from openai import OpenAI
from wayflowcore.flowbuilder import FlowBuilder
from wayflowcore.property import AnyProperty, StringProperty
from wayflowcore.steps import ParallelFlowExecutionStep
from wayflowcore.steps.step import Step, StepResult

from config import Config
from domain_config import (
    classify_document_domains,
    configured_domain_names,
    domain_ontology_entries,
    load_domain_config,
)
import tools as tool_module
from tools import setup_tools
from schema import DEFAULT_VERTEX_TYPES, DEFAULT_EDGE_TYPES


class _FilteredStartupStream:
    """Forward startup output while dropping known noisy managed-DDL warnings."""

    def __init__(self, stream, ignored_fragments: tuple[str, ...]):
        self.stream = stream
        self.ignored_fragments = ignored_fragments
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not self._should_ignore(line):
                self.stream.write(line + "\n")
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            if not self._should_ignore(self._buffer):
                self.stream.write(self._buffer)
            self._buffer = ""
        self.stream.flush()

    def _should_ignore(self, text: str) -> bool:
        return any(fragment in text for fragment in self.ignored_fragments)


@contextmanager
def _suppress_managed_schema_ddl_warnings():
    ignored = (
        "DatabaseError while executing managed schema DDL; details suppressed",
    )
    stdout = _FilteredStartupStream(sys.stdout, ignored)
    stderr = _FilteredStartupStream(sys.stderr, ignored)
    with redirect_stdout(stdout), redirect_stderr(stderr):
        yield


def _init_agent_memory(pool, config: Config):
    """Initialize Oracle AI Agent Memory SDK."""
    from oracleagentmemory.core import OracleAgentMemory
    from oracleagentmemory.core.dbschemapolicy import SchemaPolicy
    from oracleagentmemory.core.embedders import Embedder

    schema_policy = getattr(SchemaPolicy, config.memory.schema_policy)
    print(f"Agent Memory schema policy: {config.memory.schema_policy}")

    with _suppress_managed_schema_ddl_warnings():
        memory = OracleAgentMemory(
            connection=pool,
            embedder=Embedder(
                model=config.openai.embedding_provider_model,
                api_base=config.openai.base_url or None,
                api_key=config.openai.api_key,
                truncate_prompt_tokens=8192,
            ),
            table_name_prefix=config.memory.table_prefix,
            schema_policy=schema_policy,
        )
    return memory


def _seed_ontology(memory, config: Config):
    """Seed the ontology registry with default types on first run."""
    from oracleagentmemory.apis.searchscope import SearchScope

    # Check if already seeded
    results = memory.search(
        query="ontology_entry vertex_type",
        scope=SearchScope(user_id="system", agent_id="ontology_manager"),
        max_results=1,
    )
    if results:
        return  # already seeded

    vertex_types, edge_types, _active_domains = domain_ontology_entries(
        "core",
        config.ontology.domain_config_file,
    )
    if not vertex_types:
        vertex_types = DEFAULT_VERTEX_TYPES
    if not edge_types:
        edge_types = DEFAULT_EDGE_TYPES

    # Seed vertex types
    for vt in vertex_types:
        entry = json.dumps({
            "memory_type": "ontology_entry",
            "entry_type": "vertex_type",
            "label": vt["label"],
            "description": vt["description"],
            "naming_convention": vt["naming_convention"],
        })
        memory.add_memory(
            entry,
            user_id="system",
            agent_id="ontology_manager",
        )

    # Seed edge types
    for et in edge_types:
        entry = json.dumps({
            "memory_type": "ontology_entry",
            "entry_type": "edge_type",
            "label": et["label"],
            "source": et["source"],
            "target": et["target"],
        })
        memory.add_memory(
            entry,
            user_id="system",
            agent_id="ontology_manager",
        )

    print(f"  Seeded ontology: {len(vertex_types)} vertex types, "
          f"{len(edge_types)} edge types")


def _set_run_id(config: Config) -> None:
    if not config.run_id:
        config.run_id = (
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid.uuid4().hex[:6]}"
        )


def build_runtime(config: Config):
    """
    Build shared runtime dependencies.
    Returns (pool, memory).
    """
    _set_run_id(config)

    print(f"=== KG Extraction Pipeline ===")
    print(f"Run ID: {config.run_id}")
    print(f"Database: {config.db.dsn}")
    print()

    # Database connection pool
    print("Connecting to Oracle Database...")
    pool = oracledb.create_pool(
        user=config.db.user,
        password=config.db.password,
        dsn=config.db.dsn,
        min=config.db.pool_min,
        max=config.db.pool_max,
    )

    # OpenAI-compatible client. OPENAI_BASE_URL can point to Ollama or
    # another endpoint that implements the OpenAI API shape.
    openai_client = OpenAI(
        api_key=config.openai.api_key,
        base_url=config.openai.base_url or None,
        timeout=config.openai.timeout_seconds,
        max_retries=config.openai.max_retries,
    )

    # Agent Memory
    print("Initializing Agent Memory...")
    memory = _init_agent_memory(pool, config)

    # Seed ontology on first run
    print("Checking ontology registry...")
    _seed_ontology(memory, config)

    # Initialize tools with dependencies
    print("Setting up tools...")
    setup_tools(pool, openai_client, memory, config)

    return pool, memory


def _run_json_tool(tool, **kwargs) -> dict:
    result = tool.run(**kwargs)
    if isinstance(result, str):
        return json.loads(result)
    if isinstance(result, dict):
        return result
    raise TypeError(f"Tool {tool.name} returned unsupported type {type(result)}")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _safe_cache_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _ontology_signature(ontology: dict) -> str:
    entity_types = sorted(
        str(item.get("label", "")).strip().lower()
        for item in ontology.get("entity_types", [])
        if isinstance(item, dict) and str(item.get("label", "")).strip()
    )
    relationship_types = sorted(
        (
            str(item.get("label", "")).strip().lower(),
            str(item.get("source", "any") or "any").strip().lower(),
            str(item.get("target", "any") or "any").strip().lower(),
        )
        for item in ontology.get("relationship_types", [])
        if isinstance(item, dict) and str(item.get("label", "")).strip()
    )
    payload = json.dumps(
        {
            "entity_types": entity_types,
            "relationship_types": relationship_types,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _read_cached_chunk_text(config: Config, doc_fingerprint: str) -> str:
    path = Path(config.chunking.cache_dir) / f"{doc_fingerprint}.json"
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        return ""
    return "\n".join(
        str(chunk.get("text", ""))
        for chunk in chunks
        if isinstance(chunk, dict)
    )


def _determine_ontology_domains(
    config: Config,
    doc_fingerprint: str,
) -> tuple[str, dict]:
    configured = str(config.ontology.domains or "core").strip().lower()
    domain_config = load_domain_config(config.ontology.domain_config_file)
    if configured != "auto":
        domains = configured_domain_names(domain_config, configured)
        return ",".join(domains), {
            "mode": "configured",
            "document_type": next((name for name in domains if name != "core"), "generic"),
            "scores": {},
        }

    text = _read_cached_chunk_text(config, doc_fingerprint).lower()
    domains, report = classify_document_domains(domain_config, text)
    return ",".join(domains), report


def _extraction_cache_path(
    config: Config,
    doc_fingerprint: str,
    variant,
    ontology_signature: str,
) -> Path:
    cache_dir = Path(config.chunking.cache_dir).parent / "extractions"
    cache_name = "_".join([
        doc_fingerprint,
        f"ont{_safe_cache_token(ontology_signature)}",
        _safe_cache_token(variant.name),
        _safe_cache_token(variant.model),
        _safe_cache_token(variant.temperature),
        _safe_cache_token(variant.prompt_angle),
        f"bs{config.ensemble.extraction_batch_size}",
    ])
    return cache_dir / f"{cache_name}.json"


def _load_extraction_cache(
    config: Config,
    doc_fingerprint: str,
    variant,
    ontology_signature: str,
) -> dict | None:
    if not _env_flag("EXTRACTION_CACHE_ENABLED", True):
        return None
    if not _env_flag("EXTRACTION_CACHE_READ_ENABLED", True):
        return None
    if _env_flag("FORCE_REPROCESS", False):
        print(
            f"[runner] extraction cache bypassed for {variant.name}; "
            "FORCE_REPROCESS=1",
            flush=True,
        )
        return None
    path = _extraction_cache_path(
        config,
        doc_fingerprint,
        variant,
        ontology_signature,
    )
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"[runner] extraction cache hit for {variant.name}: {path}", flush=True)
    return payload


def _write_extraction_cache(
    config: Config,
    doc_fingerprint: str,
    variant,
    ontology_signature: str,
    payload: dict,
) -> None:
    if not _env_flag("EXTRACTION_CACHE_ENABLED", True):
        return
    path = _extraction_cache_path(
        config,
        doc_fingerprint,
        variant,
        ontology_signature,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[runner] extraction cache saved for {variant.name}: {path}", flush=True)


def _normalize_space(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_label(value: str, fallback: str) -> str:
    label = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    label = label.strip("_")
    return label or fallback


def _normalize_signature_endpoint(value: str) -> str:
    label = _normalize_label(value, "any")
    return "any" if label in {"", "any", "*"} else label


def _ontology_relationship_signatures(
    ontology: dict,
) -> dict[str, set[tuple[str, str]]]:
    signatures: dict[str, set[tuple[str, str]]] = {}
    for item in ontology.get("relationship_types", []):
        if not isinstance(item, dict):
            continue
        label = _normalize_label(item.get("label", ""), "")
        if not label:
            continue
        signatures.setdefault(label, set()).add((
            _normalize_signature_endpoint(item.get("source", "any")),
            _normalize_signature_endpoint(item.get("target", "any")),
        ))
    return signatures


def _relationship_signature_allowed(
    relationship: str,
    source_type: str,
    target_type: str,
    relationship_signatures: dict[str, set[tuple[str, str]]],
) -> bool:
    signatures = relationship_signatures.get(relationship)
    if not signatures:
        return True
    for allowed_source, allowed_target in signatures:
        source_matches = allowed_source == "any" or allowed_source == source_type
        target_matches = allowed_target == "any" or allowed_target == target_type
        if source_matches and target_matches:
            return True
    return False


def _name_similarity(left: str, right: str) -> float:
    left_norm = _normalize_space(left).lower()
    right_norm = _normalize_space(right).lower()
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _entity_from_triple(triple: dict, key: str) -> dict:
    entity = triple.get(key)
    if not isinstance(entity, dict):
        return {}
    name = _normalize_space(entity.get("name", ""))
    if not name:
        return {}
    properties = entity.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    return {
        "name": name,
        "type": entity.get("type", ""),
        "properties": properties,
    }


def _triples_match(left: dict, right: dict, config: Config) -> bool:
    if left["relationship"] != right["relationship"]:
        return False
    if left["subject"]["type"] != right["subject"]["type"]:
        return False
    if left["object"]["type"] != right["object"]["type"]:
        return False
    threshold = config.ensemble.name_similarity_threshold
    return (
        _name_similarity(left["subject"]["name"], right["subject"]["name"]) >= threshold
        and _name_similarity(left["object"]["name"], right["object"]["name"]) >= threshold
    )


def _ontology_label_set(ontology: dict, key: str) -> set[str]:
    labels = {
        _normalize_label(item.get("label", ""), "")
        for item in ontology.get(key, [])
        if isinstance(item, dict)
    }
    labels.discard("")
    return labels


def _pending_type_parts(value: object) -> tuple[str, str]:
    if isinstance(value, dict):
        kind = _normalize_label(
            value.get("kind") or value.get("entry_type") or value.get("type_kind") or "",
            "",
        )
        label = _normalize_label(value.get("label") or value.get("name") or "", "")
        return kind, label

    raw = str(value or "").strip()
    if ":" in raw:
        kind, label = raw.split(":", 1)
        return _normalize_label(kind, ""), _normalize_label(label, "")
    return "", _normalize_label(raw, "")


def _ontology_candidate_from_value(value: object) -> dict:
    if not isinstance(value, dict):
        kind, label = _pending_type_parts(value)
        return {
            "kind": kind,
            "label": label,
            "description": "",
            "evidence": "",
            "source": "any",
            "target": "any",
        }

    kind = _normalize_label(
        value.get("kind") or value.get("entry_type") or value.get("type_kind") or "",
        "",
    )
    if kind in {"entity", "entity_type", "vertex"}:
        kind = "vertex_type"
    elif kind in {"relationship", "relationship_type", "edge"}:
        kind = "edge_type"

    return {
        "kind": kind,
        "label": _normalize_label(value.get("label") or value.get("name") or "", ""),
        "description": _normalize_space(value.get("description", "")),
        "evidence": _normalize_space(value.get("evidence", "")),
        "source": _normalize_label(value.get("source") or "any", "any"),
        "target": _normalize_label(value.get("target") or "any", "any"),
    }


def _is_reusable_type_label(label: str) -> bool:
    if not label or len(label) < 3:
        return False
    parts = [part for part in label.split("_") if part]
    if len(parts) > 5:
        return False
    if re.fullmatch(r"[0-9_]+", label):
        return False
    value_like_tokens = {
        "patients",
        "patient",
        "year",
        "date",
        "timeline",
        "hold",
        "count",
        "million",
        "billion",
    }
    if any(part.isdigit() for part in parts):
        return False
    if len(parts) >= 3 and any(part in value_like_tokens for part in parts):
        return False
    return True


def _candidate_bucket() -> dict:
    return {
        "count": 0,
        "extractors": set(),
        "descriptions": Counter(),
        "evidence": [],
        "pairs": Counter(),
    }


def _admit_ontology_candidates(
    config: Config,
    extractor_results: dict[str, dict],
    ontology: dict,
    doc_fingerprint: str,
) -> dict:
    evidence_threshold = config.ontology.auto_admit_threshold
    extractor_threshold = config.ontology.candidate_min_extractors
    allowed_entity_types = _ontology_label_set(ontology, "entity_types")
    allowed_relationships = _ontology_label_set(ontology, "relationship_types")

    vertex_candidates: dict[str, dict] = defaultdict(_candidate_bucket)
    edge_candidates: dict[str, dict] = defaultdict(_candidate_bucket)
    ambiguous_pending: Counter[str] = Counter()
    rejected_candidates: Counter[str] = Counter()

    for extractor_name, result in extractor_results.items():
        raw_candidates = list(result.get("ontology_candidates", []))
        for pending_type in result.get("pending_types", []):
            raw_candidates.append(pending_type)

        for raw_candidate in raw_candidates:
            candidate = _ontology_candidate_from_value(raw_candidate)
            kind = candidate["kind"]
            label = candidate["label"]
            if not label or not _is_reusable_type_label(label):
                if label:
                    rejected_candidates[label] += 1
                continue
            if kind in {"vertex", "vertex_type", "entity", "entity_type"}:
                if label not in allowed_entity_types:
                    info = vertex_candidates[label]
                    info["count"] += 1
                    info["extractors"].add(extractor_name)
                    if candidate["description"]:
                        info["descriptions"][candidate["description"]] += 1
                    if candidate["evidence"]:
                        info["evidence"].append(candidate["evidence"])
            elif kind in {"edge", "edge_type", "relationship", "relationship_type"}:
                if label not in allowed_relationships:
                    source = (
                        candidate["source"]
                        if candidate["source"] in allowed_entity_types
                        else "any"
                    )
                    target = (
                        candidate["target"]
                        if candidate["target"] in allowed_entity_types
                        else "any"
                    )
                    info = edge_candidates[label]
                    info["count"] += 1
                    info["extractors"].add(extractor_name)
                    info["pairs"][(source, target)] += 1
                    if candidate["description"]:
                        info["descriptions"][candidate["description"]] += 1
                    if candidate["evidence"]:
                        info["evidence"].append(candidate["evidence"])
            elif label not in allowed_entity_types and label not in allowed_relationships:
                ambiguous_pending[label] += 1

    admitted_vertices = []
    candidate_vertices = []
    for label, info in sorted(vertex_candidates.items()):
        count = info["count"]
        extractor_count = len(info["extractors"])
        candidate_vertices.append({
            "label": label,
            "count": count,
            "extractor_count": extractor_count,
        })
        if (
            count < evidence_threshold
            or extractor_count < extractor_threshold
            or label in allowed_entity_types
        ):
            continue
        description = (
            info["descriptions"].most_common(1)[0][0]
            if info["descriptions"]
            else f"Auto-admitted from ontology candidate outputs; observed {count} times."
        )
        entry = {
            "memory_type": "ontology_entry",
            "entry_type": "vertex_type",
            "label": label,
            "description": description,
            "naming_convention": "Canonical name from source text",
            "admitted_by": "direct_runner",
            "admission_threshold": evidence_threshold,
            "observed_count": count,
            "observed_extractors": sorted(info["extractors"]),
            "extractor_threshold": extractor_threshold,
            "evidence": info["evidence"][:5],
            "run_id": config.run_id,
            "source_doc_fingerprint": doc_fingerprint,
        }
        _run_json_tool(
            tool_module.update_ontology,
            action="admit",
            entry_json=json.dumps(entry, ensure_ascii=False),
        )
        admitted_vertices.append({
            "label": label,
            "count": count,
            "extractor_count": extractor_count,
        })
        allowed_entity_types.add(label)

    admitted_edges = []
    candidate_edges = []
    for label, info in sorted(edge_candidates.items()):
        count = info["count"]
        extractor_count = len(info["extractors"])
        pair_counts = info["pairs"]
        if pair_counts:
            (source_type, target_type), _pair_count = pair_counts.most_common(1)[0]
        else:
            source_type, target_type = "any", "any"
        candidate_edges.append({
            "label": label,
            "count": count,
            "extractor_count": extractor_count,
            "source": source_type,
            "target": target_type,
        })
        if (
            count < evidence_threshold
            or extractor_count < extractor_threshold
            or label in allowed_relationships
        ):
            continue
        description = (
            info["descriptions"].most_common(1)[0][0]
            if info["descriptions"]
            else f"Auto-admitted from ontology candidate outputs; observed {count} times."
        )
        entry = {
            "memory_type": "ontology_entry",
            "entry_type": "edge_type",
            "label": label,
            "source": source_type,
            "target": target_type,
            "description": description,
            "admitted_by": "direct_runner",
            "admission_threshold": evidence_threshold,
            "observed_count": count,
            "observed_extractors": sorted(info["extractors"]),
            "extractor_threshold": extractor_threshold,
            "evidence": info["evidence"][:5],
            "run_id": config.run_id,
            "source_doc_fingerprint": doc_fingerprint,
        }
        _run_json_tool(
            tool_module.update_ontology,
            action="admit",
            entry_json=json.dumps(entry, ensure_ascii=False),
        )
        admitted_edges.append({
            "label": label,
            "count": count,
            "source": source_type,
            "target": target_type,
        })
        allowed_relationships.add(label)

    if admitted_vertices or admitted_edges:
        print(
            "[runner] ontology auto-admit completed; "
            f"evidence_threshold={evidence_threshold}, "
            f"extractor_threshold={extractor_threshold}, "
            f"vertex_types={admitted_vertices}, edge_types={admitted_edges}",
            flush=True,
        )
    elif candidate_vertices or candidate_edges or ambiguous_pending or rejected_candidates:
        print(
            "[runner] ontology auto-admit skipped; "
            f"evidence_threshold={evidence_threshold}, "
            f"extractor_threshold={extractor_threshold}, "
            f"candidate_vertex_types={candidate_vertices}, "
            f"candidate_edge_types={candidate_edges}, "
            f"ambiguous_pending_types={dict(ambiguous_pending)}, "
            f"rejected_candidates={dict(rejected_candidates)}",
            flush=True,
        )

    return {
        "vertices": admitted_vertices,
        "edges": admitted_edges,
        "candidate_vertices": candidate_vertices,
        "candidate_edges": candidate_edges,
        "ambiguous_pending_types": dict(ambiguous_pending),
        "rejected_candidates": dict(rejected_candidates),
    }


def _canonical_name(name: str, entity_type: str) -> str:
    name = _normalize_space(name)
    if entity_type in {"person", "organization", "location", "event", "document"}:
        return name.title() if not name.isupper() else name
    return name


def _vertex_id(name: str, entity_type: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    raw = f"{entity_type}:{slug or 'entity'}"
    if len(raw) <= 190:
        return raw
    digest = uuid.uuid5(uuid.NAMESPACE_URL, raw).hex[:16]
    return f"{entity_type}:{slug[:160]}_{digest}"


def _record_sort_key(record: dict) -> tuple:
    return (
        int(record.get("source_chunk_index", -1)),
        str(record.get("relationship", "")),
        str(record.get("subject", {}).get("type", "")),
        _normalize_space(record.get("subject", {}).get("name", "")).lower(),
        str(record.get("object", {}).get("type", "")),
        _normalize_space(record.get("object", {}).get("name", "")).lower(),
        str(record.get("extractor", "")),
    )


def _resolve_entity_against_graph(
    entity: dict,
    cache: dict[tuple[str, str], dict],
) -> dict:
    entity_type = entity["type"]
    canonical = _canonical_name(entity["name"], entity_type)
    cache_key = (canonical, entity_type)
    if cache_key in cache:
        return cache[cache_key]

    matches = _run_json_tool(
        tool_module.lookup_similar_vertices,
        entity_name=canonical,
        entity_type=entity_type,
    ).get("similar_vertices", [])
    if matches:
        best = matches[0]
        resolved = {
            "vertex_id": best["vertex_id"],
            "canonical_name": best["canonical_name"],
            "vertex_type": best.get("vertex_type", entity_type) or entity_type,
            "matched_existing": True,
            "similarity": best.get("similarity"),
        }
    else:
        resolved = {
            "vertex_id": _vertex_id(canonical, entity_type),
            "canonical_name": canonical,
            "vertex_type": entity_type,
            "matched_existing": False,
            "similarity": None,
        }
    cache[cache_key] = resolved
    return resolved


def _consensus_triples(
    extractor_results: dict[str, dict],
    ontology: dict,
    config: Config,
) -> tuple[list[dict], int]:
    preferred_extractor = (
        config.ensemble.variants[0].name
        if config.ensemble.variants else ""
    )
    allowed_entity_types = {
        _normalize_label(item.get("label", ""), "")
        for item in ontology.get("entity_types", [])
        if isinstance(item, dict)
    }
    allowed_relationships = {
        _normalize_label(item.get("label", ""), "")
        for item in ontology.get("relationship_types", [])
        if isinstance(item, dict)
    }
    relationship_signatures = _ontology_relationship_signatures(ontology)
    allowed_entity_types.discard("")
    allowed_relationships.discard("")
    strict_ontology = bool(getattr(config.ensemble, "strict_ontology", True))

    records = []
    for extractor_name, result in extractor_results.items():
        for triple in result.get("triples", []):
            if not isinstance(triple, dict):
                continue
            subject = _entity_from_triple(triple, "subject")
            obj = _entity_from_triple(triple, "object")
            if not subject or not obj:
                continue

            subject["type"] = _normalize_label(subject["type"], "concept")
            obj["type"] = _normalize_label(obj["type"], "concept")
            relationship = _normalize_label(triple.get("relationship", ""), "related_to")

            if strict_ontology and (
                (allowed_entity_types and subject["type"] not in allowed_entity_types)
                or (allowed_entity_types and obj["type"] not in allowed_entity_types)
                or (allowed_relationships and relationship not in allowed_relationships)
                or not _relationship_signature_allowed(
                    relationship,
                    subject["type"],
                    obj["type"],
                    relationship_signatures,
                )
            ):
                continue

            if allowed_entity_types and subject["type"] not in allowed_entity_types:
                subject["type"] = "concept"
            if allowed_entity_types and obj["type"] not in allowed_entity_types:
                obj["type"] = "concept"
            if allowed_relationships and relationship not in allowed_relationships:
                relationship = "related_to"

            try:
                source_chunk_index = int(triple.get("source_chunk_index", -1))
            except (TypeError, ValueError):
                source_chunk_index = -1
            try:
                confidence = float(triple.get("confidence", 0.8) or 0.8)
            except (TypeError, ValueError):
                confidence = 0.8

            records.append({
                "extractor": extractor_name,
                "subject": subject,
                "relationship": relationship,
                "object": obj,
                "confidence": confidence,
                "source_chunk_index": source_chunk_index,
                "source_doc": triple.get("source_doc", ""),
            })

    records.sort(key=_record_sort_key)

    groups: list[dict] = []
    for record in records:
        match = next(
            (
                group for group in groups
                if _triples_match(group["representative"], record, config)
            ),
            None,
        )
        if match is None:
            groups.append({
                "representative": record,
                "members": [record],
                "extractors": {record["extractor"]},
            })
        else:
            match["members"].append(record)
            match["extractors"].add(record["extractor"])
            if record["extractor"] == preferred_extractor:
                match["representative"] = record

    accepted = []
    for group in groups:
        consensus_count = len(group["extractors"])
        if consensus_count < config.ensemble.consensus_k:
            continue
        representative = group["representative"]
        accepted.append({
            **representative,
            "consensus_count": consensus_count,
            "agreeing_extractors": sorted(group["extractors"]),
            "confidence": consensus_count / max(1, config.ensemble.consensus_n),
        })

    accepted.sort(key=_record_sort_key)
    return accepted, len(records)


def _write_consensus_to_graph(
    config: Config,
    doc_manifest: dict,
    ontology_json: str,
    consensus: list[dict],
    raw_triple_count: int,
) -> dict:
    print("[graphwriter] creating/validating typed graph DDL", flush=True)
    _run_json_tool(tool_module.create_graph_ddl, ontology_json=ontology_json)

    vertices = {}
    edges = {}
    entity_cache: dict[tuple[str, str], dict] = {}
    doc_fingerprint = str(doc_manifest.get("doc_fingerprint", ""))
    for triple in consensus:
        source_doc = triple.get("source_doc") or doc_manifest.get("filename", "")
        source_chunk = int(triple.get("source_chunk_index", -1))
        resolved: dict[str, dict] = {}
        for role in ("subject", "object"):
            entity = triple[role]
            resolved[role] = _resolve_entity_against_graph(entity, entity_cache)
            vertex_key = (
                resolved[role]["canonical_name"],
                resolved[role]["vertex_type"],
            )
            vertices[vertex_key] = {
                "vertex_id": resolved[role]["vertex_id"],
                "canonical_name": resolved[role]["canonical_name"],
                "vertex_type": resolved[role]["vertex_type"],
                "properties": entity.get("properties", {}),
                "doc_fingerprint": doc_fingerprint,
                "source_doc": source_doc,
                "source_chunk": source_chunk,
                "confidence": triple["confidence"],
                "consensus_count": triple["consensus_count"],
            }

        source_vertex_id = resolved["subject"]["vertex_id"]
        target_vertex_id = resolved["object"]["vertex_id"]
        edges[(source_vertex_id, triple["relationship"], target_vertex_id)] = {
            "source_vertex_id": source_vertex_id,
            "target_vertex_id": target_vertex_id,
            "relationship_type": triple["relationship"],
            "doc_fingerprint": doc_fingerprint,
            "source_doc": source_doc,
            "source_chunk": source_chunk,
            "confidence": triple["confidence"],
            "consensus_count": triple["consensus_count"],
        }

    vertices_created = 0
    vertices_updated = 0
    for vertex in vertices.values():
        result = _run_json_tool(
            tool_module.merge_vertex,
            vertex_id=vertex["vertex_id"],
            canonical_name=vertex["canonical_name"],
            vertex_type=vertex["vertex_type"],
            properties_json=json.dumps(vertex["properties"], ensure_ascii=False),
            doc_fingerprint=vertex["doc_fingerprint"],
            source_doc=vertex["source_doc"],
            source_chunk=vertex["source_chunk"],
            confidence=vertex["confidence"],
            consensus_count=vertex["consensus_count"],
        )
        if result.get("action") == "updated":
            vertices_updated += 1
        else:
            vertices_created += 1

    for edge in edges.values():
        _run_json_tool(
            tool_module.merge_edge,
            source_vertex_id=edge["source_vertex_id"],
            target_vertex_id=edge["target_vertex_id"],
            relationship_type=edge["relationship_type"],
            properties_json=json.dumps({}, ensure_ascii=False),
            doc_fingerprint=edge["doc_fingerprint"],
            source_doc=edge["source_doc"],
            source_chunk=edge["source_chunk"],
            confidence=edge["confidence"],
            consensus_count=edge["consensus_count"],
        )

    _run_json_tool(
        tool_module.refresh_typed_graph_views,
        ontology_json=ontology_json,
    )

    chunks_result = _run_json_tool(
        tool_module.store_chunks,
        chunks_json=json.dumps({"doc_fingerprint": doc_manifest["doc_fingerprint"]}),
    )

    consensus_rate = (
        len(consensus) / raw_triple_count
        if raw_triple_count else 0.0
    )
    _run_json_tool(
        tool_module.save_extraction_record,
        doc_fingerprint=doc_manifest["doc_fingerprint"],
        filename=doc_manifest.get("filename", ""),
        chunks_processed=int(doc_manifest.get("chunk_count", 0)),
        vertices_created=vertices_created,
        edges_created=len(edges),
        consensus_rate=consensus_rate,
    )

    return {
        "vertices_created": vertices_created,
        "vertices_updated": vertices_updated,
        "edges_written": len(edges),
        "chunks_stored": chunks_result.get("chunks_stored", 0),
        "consensus_rate": consensus_rate,
    }


class GraphExtractionStep(Step):
    """WayFlow step wrapper around one deterministic pipeline phase."""

    _input_descriptors_change_step_behavior = True
    _output_descriptors_change_step_behavior = True

    def __init__(
        self,
        handler: Callable[[dict[str, Any]], dict[str, Any]],
        input_descriptors=None,
        output_descriptors=None,
        input_mapping=None,
        output_mapping=None,
        name: str | None = None,
        __metadata_info__=None,
    ):
        self.handler = handler
        super().__init__(
            step_static_configuration={},
            input_descriptors=input_descriptors,
            output_descriptors=output_descriptors,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            name=name,
            __metadata_info__=__metadata_info__,
        )

    @classmethod
    def _get_step_specific_static_configuration_descriptors(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def _compute_step_specific_input_descriptors_from_static_config(
        cls,
        input_descriptors,
        output_descriptors,
    ):
        return input_descriptors or []

    @classmethod
    def _compute_step_specific_output_descriptors_from_static_config(
        cls,
        input_descriptors,
        output_descriptors,
    ):
        return output_descriptors or []

    @property
    def might_yield(self) -> bool:
        return False

    def _invoke_step(self, inputs: dict[str, Any], conversation) -> StepResult:
        return StepResult(outputs=self.handler(inputs))

    async def _invoke_step_async(
        self,
        inputs: dict[str, Any],
        conversation,
    ) -> StepResult:
        return await anyio.to_thread.run_sync(
            self._invoke_step,
            inputs,
            conversation,
        )


def _build_document_flow(config: Config):
    parser = GraphExtractionStep(
        name="ParserAgent",
        input_descriptors=[
            StringProperty(
                name="document_path",
                description="Path of the document to parse",
            )
        ],
        output_descriptors=[
            AnyProperty(name="manifest"),
            StringProperty(name="doc_fingerprint"),
        ],
        handler=lambda inputs: _step_parse_document(config, inputs),
    )
    document_type = GraphExtractionStep(
        name="DocumentTypeAgent",
        input_descriptors=[
            AnyProperty(name="manifest"),
            StringProperty(name="doc_fingerprint"),
        ],
        output_descriptors=[
            AnyProperty(name="manifest"),
            StringProperty(name="doc_fingerprint"),
            StringProperty(name="domain_names"),
            AnyProperty(name="domain_report"),
        ],
        handler=lambda inputs: _step_classify_document(config, inputs),
    )
    ontology = GraphExtractionStep(
        name="OntologyAgent",
        input_descriptors=[
            AnyProperty(name="manifest"),
            StringProperty(name="doc_fingerprint"),
            StringProperty(name="domain_names"),
            AnyProperty(name="domain_report"),
        ],
        output_descriptors=[
            AnyProperty(name="manifest"),
            StringProperty(name="doc_fingerprint"),
            StringProperty(name="domain_names"),
            AnyProperty(name="domain_report"),
            AnyProperty(name="ontology"),
            StringProperty(name="ontology_json"),
            StringProperty(name="ontology_signature"),
            AnyProperty(name="cache_status"),
            AnyProperty(name="skip_result"),
            StringProperty(name="should_skip"),
        ],
        handler=lambda inputs: _step_load_ontology_and_cache(config, inputs),
    )
    extractor_result_inputs = [
        AnyProperty(name=f"{variant.name}_result")
        for variant in config.ensemble.variants
    ]
    extractor_parallel = ParallelFlowExecutionStep(
        flows=[
            _build_extractor_subflow(config, variant.name)
            for variant in config.ensemble.variants
        ],
        max_workers=config.ensemble.parallel_workers,
        name="ParallelExtractorAgents",
    )
    admit = GraphExtractionStep(
        name="OntologyAdmitTask",
        input_descriptors=[
            AnyProperty(name="manifest"),
            StringProperty(name="doc_fingerprint"),
            StringProperty(name="domain_names"),
            AnyProperty(name="domain_report"),
            AnyProperty(name="ontology"),
            StringProperty(name="ontology_json"),
            AnyProperty(name="cache_status"),
            AnyProperty(name="skip_result"),
            StringProperty(name="should_skip"),
            *extractor_result_inputs,
        ],
        output_descriptors=[
            AnyProperty(name="manifest"),
            StringProperty(name="doc_fingerprint"),
            StringProperty(name="domain_names"),
            AnyProperty(name="domain_report"),
            AnyProperty(name="ontology"),
            StringProperty(name="ontology_json"),
            AnyProperty(name="cache_status"),
            AnyProperty(name="skip_result"),
            StringProperty(name="should_skip"),
            AnyProperty(name="extractor_results"),
            AnyProperty(name="ontology_updates"),
        ],
        handler=lambda inputs: _step_admit_ontology(config, inputs),
    )
    consensus = GraphExtractionStep(
        name="ConsensusMergerTask",
        input_descriptors=[
            AnyProperty(name="manifest"),
            StringProperty(name="domain_names"),
            AnyProperty(name="domain_report"),
            AnyProperty(name="ontology"),
            StringProperty(name="ontology_json"),
            AnyProperty(name="skip_result"),
            StringProperty(name="should_skip"),
            AnyProperty(name="extractor_results"),
            AnyProperty(name="ontology_updates"),
        ],
        output_descriptors=[
            AnyProperty(name="manifest"),
            StringProperty(name="domain_names"),
            AnyProperty(name="domain_report"),
            StringProperty(name="ontology_json"),
            AnyProperty(name="skip_result"),
            StringProperty(name="should_skip"),
            AnyProperty(name="extractor_results"),
            AnyProperty(name="ontology_updates"),
            AnyProperty(name="consensus"),
            AnyProperty(name="raw_triple_count"),
        ],
        handler=lambda inputs: _step_consensus(config, inputs),
    )
    writer = GraphExtractionStep(
        name="GraphWriterTask",
        input_descriptors=[
            AnyProperty(name="manifest"),
            StringProperty(name="domain_names"),
            AnyProperty(name="domain_report"),
            StringProperty(name="ontology_json"),
            AnyProperty(name="skip_result"),
            StringProperty(name="should_skip"),
            AnyProperty(name="extractor_results"),
            AnyProperty(name="ontology_updates"),
            AnyProperty(name="consensus"),
            AnyProperty(name="raw_triple_count"),
        ],
        output_descriptors=[AnyProperty(name="result")],
        handler=lambda inputs: _step_write_graph(config, inputs),
    )

    builder = FlowBuilder().add_sequence([
        parser,
        document_type,
        ontology,
        extractor_parallel,
        admit,
        consensus,
        writer,
    ])
    return (
        builder
        .set_entry_point(parser)
        .set_finish_points(writer, output_descriptors=[AnyProperty(name="result")])
        .build(
            name="KGExtractionAgentFlow",
            description=(
                "WayFlow step flow for parser, document type detection, "
                "parallel extractors, consensus, and graph writer."
            ),
        )
    )


def _build_extractor_subflow(config: Config, variant_name: str):
    step_label = _extractor_step_label(variant_name)
    result_descriptor = AnyProperty(name=f"{variant_name}_result")
    step = GraphExtractionStep(
        name=f"{step_label}Agent",
        input_descriptors=[
            AnyProperty(name="manifest"),
            StringProperty(name="doc_fingerprint"),
            StringProperty(name="ontology_json"),
            StringProperty(name="ontology_signature"),
            AnyProperty(name="cache_status"),
            AnyProperty(name="skip_result"),
            StringProperty(name="should_skip"),
        ],
        output_descriptors=[result_descriptor],
        handler=lambda inputs: _step_extract_variant(config, variant_name, inputs),
    )
    return (
        FlowBuilder()
        .add_sequence([step])
        .set_entry_point(step)
        .set_finish_points(step, output_descriptors=[result_descriptor])
        .build(
            name=f"{step_label}Flow",
            description=f"Single extractor sub-flow for {variant_name}.",
        )
    )


def _extractor_step_label(variant_name: str) -> str:
    if variant_name.startswith("extractor_"):
        suffix = variant_name.removeprefix("extractor_")
        return f"Extractor{suffix}"
    return "Extractor" + "".join(
        part.capitalize()
        for part in variant_name.replace("-", "_").split("_")
        if part
    )


def _step_parse_document(config: Config, inputs: dict[str, Any]) -> dict[str, Any]:
    print("Parsing document...", flush=True)
    manifest = _run_json_tool(tool_module.parse_document, file_path=inputs["document_path"])
    return {
        "manifest": manifest,
        "doc_fingerprint": manifest["doc_fingerprint"],
    }


def _step_classify_document(config: Config, inputs: dict[str, Any]) -> dict[str, Any]:
    doc_fingerprint = inputs["doc_fingerprint"]
    domain_names, domain_report = _determine_ontology_domains(config, doc_fingerprint)
    print(
        "[runner] document type evaluated; "
        f"type={domain_report.get('document_type')}, "
        f"domains={domain_names}, mode={domain_report.get('mode')}",
        flush=True,
    )
    return {
        **inputs,
        "domain_names": domain_names,
        "domain_report": domain_report,
    }


def _step_load_ontology_and_cache(config: Config, inputs: dict[str, Any]) -> dict[str, Any]:
    manifest = inputs["manifest"]
    doc_fingerprint = inputs["doc_fingerprint"]
    domain_names = inputs["domain_names"]
    domain_report = inputs["domain_report"]
    print(
        f"[runner] ontology domains selected: {domain_names}; "
        f"mode={domain_report.get('mode')}",
        flush=True,
    )

    ontology = _run_json_tool(
        tool_module.read_ontology,
        exclude_doc_fingerprint=doc_fingerprint,
        domain_names=domain_names,
    )
    ontology_json = json.dumps(ontology, ensure_ascii=False)
    ontology_sig = _ontology_signature(ontology)
    cache_status = _run_json_tool(
        tool_module.check_extraction_cache,
        doc_fingerprint=doc_fingerprint,
    )

    force_reprocess = _env_flag("FORCE_REPROCESS", False)
    should_skip = cache_status.get("already_processed") and not force_reprocess
    skip_result = None
    if should_skip:
        print(
            "[graphwriter] creating/validating typed graph DDL for cached extraction",
            flush=True,
        )
        _run_json_tool(tool_module.create_graph_ddl, ontology_json=ontology_json)
        print(
            "Document already has an extraction record; skipping. "
            "Set FORCE_REPROCESS=1 to run it again.",
            flush=True,
        )
        skip_result = {
            "skipped": True,
            "manifest": manifest,
            "ontology_domains": domain_names.split(","),
            "ontology_domain_report": domain_report,
        }

    return {
        "manifest": manifest,
        "doc_fingerprint": doc_fingerprint,
        "domain_names": domain_names,
        "domain_report": domain_report,
        "ontology": ontology,
        "ontology_json": ontology_json,
        "ontology_signature": ontology_sig,
        "cache_status": cache_status,
        "skip_result": skip_result,
        "should_skip": "true" if should_skip else "false",
    }


def _step_extract_variant(
    config: Config,
    variant_name: str,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    if inputs.get("should_skip") == "true":
        return {f"{variant_name}_result": {}}

    doc_fingerprint = inputs["doc_fingerprint"]
    manifest = inputs["manifest"]
    ontology_sig = inputs["ontology_signature"]
    variant = next(
        variant for variant in config.ensemble.variants
        if variant.name == variant_name
    )
    cached_result = _load_extraction_cache(
        config,
        doc_fingerprint,
        variant,
        ontology_sig,
    )
    if cached_result is not None:
        return {f"{variant_name}_result": cached_result}

    print(
        f"[runner] extractor {variant.name} started; "
        f"model={variant.model}, batch_size={config.ensemble.extraction_batch_size}",
        flush=True,
    )
    result = _run_json_tool(
        tool_module.extract_document_triples,
        doc_fingerprint=doc_fingerprint,
        model=variant.model,
        temperature=variant.temperature,
        prompt_angle=variant.prompt_angle,
        ontology_json=inputs["ontology_json"],
        source_doc=manifest.get("filename", ""),
    )
    print(
        f"[runner] extractor {variant.name} completed; "
        f"triples={len(result.get('triples', []))}",
        flush=True,
    )
    _write_extraction_cache(
        config=config,
        doc_fingerprint=doc_fingerprint,
        variant=variant,
        ontology_signature=ontology_sig,
        payload=result,
    )
    return {f"{variant_name}_result": result}


def _step_admit_ontology(config: Config, inputs: dict[str, Any]) -> dict[str, Any]:
    if inputs.get("should_skip") == "true":
        return {
            **inputs,
            "extractor_results": {},
            "ontology_updates": {
                "vertices": [],
                "edges": [],
                "candidate_vertices": [],
                "candidate_edges": [],
                "ambiguous_pending_types": {},
                "rejected_candidates": {},
            },
        }

    extractor_results = {
        variant.name: inputs.get(f"{variant.name}_result", {})
        for variant in config.ensemble.variants
    }
    ontology_updates = _admit_ontology_candidates(
        config=config,
        extractor_results=extractor_results,
        ontology=inputs["ontology"],
        doc_fingerprint=inputs["doc_fingerprint"],
    )
    if ontology_updates["vertices"] or ontology_updates["edges"]:
        print(
            "[runner] ontology updates saved for future documents; "
            "current run keeps its initial ontology snapshot",
            flush=True,
        )
    return {
        **inputs,
        "extractor_results": extractor_results,
        "ontology_updates": ontology_updates,
    }


def _step_consensus(config: Config, inputs: dict[str, Any]) -> dict[str, Any]:
    if inputs.get("should_skip") == "true":
        return {**inputs, "consensus": [], "raw_triple_count": 0}

    consensus, raw_triple_count = _consensus_triples(
        extractor_results=inputs["extractor_results"],
        ontology=inputs["ontology"],
        config=config,
    )
    print(
        f"[runner] consensus completed; accepted={len(consensus)}, "
        f"raw_triples={raw_triple_count}",
        flush=True,
    )
    return {
        **inputs,
        "consensus": consensus,
        "raw_triple_count": raw_triple_count,
    }


def _step_write_graph(config: Config, inputs: dict[str, Any]) -> dict[str, Any]:
    if inputs.get("should_skip") == "true":
        return {"result": inputs["skip_result"]}

    write_report = _write_consensus_to_graph(
        config=config,
        doc_manifest=inputs["manifest"],
        ontology_json=inputs["ontology_json"],
        consensus=inputs["consensus"],
        raw_triple_count=inputs["raw_triple_count"],
    )
    print(f"[runner] graph write completed; {write_report}", flush=True)
    result = {
        "skipped": False,
        "manifest": inputs["manifest"],
        "ontology_domains": inputs["domain_names"].split(","),
        "ontology_domain_report": inputs["domain_report"],
        "extractors": {
            name: {
                "triples": len(result.get("triples", [])),
                "batches": result.get("batches_processed", 0),
                "chunks": result.get("chunks_processed", 0),
            }
            for name, result in inputs["extractor_results"].items()
        },
        "raw_triples": inputs["raw_triple_count"],
        "consensus_triples": len(inputs["consensus"]),
        "ontology_updates": inputs["ontology_updates"],
        "write_report": write_report,
    }
    return {"result": result}


def _run_document_flow(config: Config, flow, doc_path: str) -> dict:
    conversation = flow.start_conversation(inputs={"document_path": doc_path})
    status = conversation.execute()
    output_values = getattr(status, "output_values", {})
    result = output_values.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"WayFlow execution did not return a result: {status}")
    return result


def run_extraction_wayflow(config: Config, document_paths: list[str]):
    """
    Run the KG extraction pipeline through a WayFlow step graph.
    """
    pool, _memory = build_runtime(config)

    try:
        flow = _build_document_flow(config)
        for doc_path in document_paths:
            print(f"{'='*60}")
            print(f"Processing: {doc_path}")
            print(f"{'='*60}")
            print("Executing WayFlow agent step flow...", flush=True)
            result = _run_document_flow(config, flow, doc_path)
            print(f"\nPipeline result:\n{json.dumps(result, indent=2)}\n")

        print(f"\n{'='*60}")
        print(f"Extraction run {config.run_id} complete.")
        print(f"{'='*60}")

    finally:
        pool.close()


def run_extraction(config: Config, document_paths: list[str]):
    """
    Run the KG extraction pipeline on a list of documents.
    """
    run_extraction_wayflow(config, document_paths)


# ══════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════

def main():
    """CLI entry point."""
    config = Config()

    if len(sys.argv) < 2:
        print("Usage: python -m runner <document_path> [doc2] [doc3] ...")
        print()
        print("Environment variables:")
        print("  ORACLE_USER      - Database user (default: kg_swarm_user)")
        print("  ORACLE_PASSWORD  - Database password")
        print("  ORACLE_DSN       - Database DSN (default: localhost:1521/FREEPDB1)")
        print("  OPENAI_API_KEY   - OpenAI API key (required)")
        print("  OPENAI_BASE_URL  - Optional OpenAI-compatible API base URL")
        print("  EXTRACTOR_COUNT - Number of parallel extractors (default: 3)")
        print("  EXTRACTOR_MODELS - Comma-separated model list, one per extractor")
        print("  EXTRACTOR_TEMPERATURES - Comma-separated temperature list")
        print("  EXTRACTOR_PROMPT_ANGLES - Optional comma-separated prompt angle list")
        print("  EXTRACTION_BATCH_SIZE - Chunks per extraction LLM call (default: 30)")
        print("  CONSENSUS_MIN_AGREEMENT - Extractor agreement count (default: 2)")
        print("  FORCE_REPROCESS - Set to 1 to ignore prior extraction records")
        print("  EXTRACTION_CACHE_ENABLED - Cache extractor outputs locally (default: true)")
        print("  DB_OBJECT_PREFIX - Optional DB object prefix (example: DEV_)")
        print()
        print("Env file:")
        print("  The app reads .env from the current directory, then this runner directory.")
        print("  Set GRAPH_SWARM_ENV_FILE to load a specific env file path.")
        sys.exit(1)

    document_paths = sys.argv[1:]

    # Validate files exist
    for path in document_paths:
        if not Path(path).exists():
            print(f"Error: file not found: {path}")
            sys.exit(1)

    run_extraction(config, document_paths)


if __name__ == "__main__":
    main()
