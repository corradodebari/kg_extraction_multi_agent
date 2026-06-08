"""
Tool implementations used by the WayFlow knowledge-graph extraction flow.
"""

import json
import hashlib
import re
import time
import unicodedata
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from uuid import uuid4

import oracledb
from openai import OpenAI
from wayflowcore.tools import tool

from config import Config
from domain_config import domain_ontology_entries
from models import (
    Chunk, ChunkMetadata, Triple, Entity,
    ExtractorResult, ConsensusTriple, ReconciledTriple,
    WriteReport, ValidationIssue,
)


# ── Globals (initialized by setup_tools) ─────────────────────
_pool: oracledb.ConnectionPool = None
_openai: OpenAI = None
_memory = None  # OracleAgentMemory instance
_config: Config = None
_document_chunk_cache: dict[str, list[dict]] = {}


def setup_tools(pool, openai_client, memory, config: Config):
    """Initialize module-level dependencies. Called once at startup."""
    global _pool, _openai, _memory, _config
    _pool = pool
    _openai = openai_client
    _memory = memory
    _config = config
    if _config.chunking.cache_enabled:
        Path(_config.chunking.cache_dir).mkdir(parents=True, exist_ok=True)
        _parser_log(f"chunk file cache directory: {_config.chunking.cache_dir}")
    else:
        _parser_log("chunk file cache disabled")


# ── Embedding helper ─────────────────────────────────────────

def _embed(text: str) -> list[float]:
    """Generate embedding via OpenAI API."""
    resp = _openai.embeddings.create(
        model=_config.openai.embedding_api_model,
        input=text,
        dimensions=_config.openai.embedding_dimensions,
    )
    return resp.data[0].embedding


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Batch embedding for efficiency."""
    if not texts:
        return []
    resp = _openai.embeddings.create(
        model=_config.openai.embedding_api_model,
        input=texts,
        dimensions=_config.openai.embedding_dimensions,
    )
    return [d.embedding for d in resp.data]


def _vector_literal(embedding: list[float]) -> str:
    """Format a Python float list for Oracle TO_VECTOR(:bind, dim, FLOAT32)."""
    return "[" + ",".join(str(float(value)) for value in embedding) + "]"


def _set_clob_inputs(cur, *bind_names: str) -> None:
    """Tell python-oracledb to bind large JSON/vector text as CLOB."""
    if bind_names:
        cur.setinputsizes(**{name: oracledb.DB_TYPE_CLOB for name in bind_names})


def _build_docling_converter(use_ocr: bool):
    """Create a Docling converter that keeps PDF OCR opt-in."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf_options = PdfPipelineOptions(do_ocr=use_ocr)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        }
    )


def _tool_log(component: str, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{component} {timestamp}] {message}", flush=True)


def _parser_log(message: str) -> None:
    _tool_log("parser", message)


def _is_already_exists_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "already exists" in text or "name is already used" in text


def _is_missing_object_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "does not exist" in text or "ora-12003" in text or "ora-42421" in text


def _agent_memory_ontology_only() -> bool:
    """Whether Agent Memory should be used only for ontology entries."""
    return bool(_config and _config.memory.ontology_only)


def _property_graph_has_identity_properties(cur, graph_name: str) -> bool:
    """Check whether SQL Developer can project graph element IDs."""
    cur.execute("""
        SELECT COUNT(DISTINCT property_name)
        FROM user_pg_label_properties
        WHERE graph_name = :graph_name
          AND (
            (label_name = 'entity' AND property_name = 'VERTEX_ID')
            OR (label_name = 'relationship' AND property_name = 'EDGE_ID')
          )
    """, {"graph_name": graph_name.upper()})
    return cur.fetchone()[0] == 2


def _property_graph_has_typed_labels(
    cur,
    graph_name: str,
    vertex_types: list[dict],
    edge_types: list[dict],
) -> bool:
    expected = {
        str(item.get("label", "")).strip().lower()
        for item in vertex_types + edge_types
        if str(item.get("label", "")).strip()
    }
    if not expected:
        return False

    cur.execute("""
        SELECT DISTINCT LOWER(label_name)
        FROM user_pg_label_properties
        WHERE graph_name = :graph_name
    """, {"graph_name": graph_name.upper()})
    found = {row[0] for row in cur.fetchall()}
    return expected.issubset(found)


def _drop_materialized_views(cur, names: list[str]) -> None:
    for name in reversed(names):
        try:
            cur.execute(f"DROP MATERIALIZED VIEW {name}")
        except oracledb.DatabaseError as exc:
            if not _is_missing_object_error(exc):
                raise


def _create_materialized_views(cur, ddls: list[str]) -> None:
    for ddl in ddls:
        try:
            cur.execute(ddl)
        except oracledb.DatabaseError as exc:
            if not _is_already_exists_error(exc):
                raise


def _cumulative_ontology_entries(
    cur,
    vertex_types: list[dict],
    edge_types: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Merge current ontology entries with labels already stored in the graph."""
    vertices_by_label: dict[str, dict] = {}
    for item in vertex_types:
        if not isinstance(item, dict):
            continue
        label = _normalize_extracted_label(item.get("label", ""))
        if label:
            vertices_by_label[label] = {**item, "label": label}

    try:
        cur.execute(f"""
            SELECT DISTINCT vertex_type
            FROM {_config.vertex_table}
            WHERE vertex_type IS NOT NULL
        """)
        for (label_raw,) in cur.fetchall():
            label = _normalize_extracted_label(label_raw)
            if label and label not in vertices_by_label:
                vertices_by_label[label] = {
                    "label": label,
                    "description": "Observed in cumulative graph.",
                    "naming_convention": "",
                }
    except oracledb.DatabaseError as exc:
        if not _is_missing_object_error(exc):
            raise

    edges_by_signature: dict[tuple[str, str, str], dict] = {}
    for item in edge_types:
        if not isinstance(item, dict):
            continue
        label = _normalize_extracted_label(item.get("label", ""))
        if not label:
            continue
        source = _normalize_signature_endpoint(item.get("source", "any"))
        target = _normalize_signature_endpoint(item.get("target", "any"))
        edges_by_signature[(label, source, target)] = {
            **item,
            "label": label,
            "source": source,
            "target": target,
        }

    try:
        cur.execute(f"""
            SELECT DISTINCT e.relationship_type, sv.vertex_type, tv.vertex_type
            FROM {_config.edge_table} e
            JOIN {_config.vertex_table} sv
              ON sv.vertex_id = e.source_vertex_id
            JOIN {_config.vertex_table} tv
              ON tv.vertex_id = e.target_vertex_id
            WHERE e.relationship_type IS NOT NULL
        """)
        for label_raw, source_raw, target_raw in cur.fetchall():
            label = _normalize_extracted_label(label_raw)
            source = _normalize_signature_endpoint(source_raw)
            target = _normalize_signature_endpoint(target_raw)
            if not label:
                continue
            edges_by_signature.setdefault(
                (label, source, target),
                {
                    "label": label,
                    "source": source,
                    "target": target,
                    "description": "Observed in cumulative graph.",
                },
            )
    except oracledb.DatabaseError as exc:
        if not _is_missing_object_error(exc):
            raise

    return (
        [vertices_by_label[label] for label in sorted(vertices_by_label)],
        [
            edges_by_signature[key]
            for key in sorted(edges_by_signature)
        ],
    )


def _chunk_document_with_docling(file_path: str, use_ocr: bool) -> list[str]:
    """Chunk a document using the GraphRAG builder's Docling chunking path."""
    from docling.chunking import HierarchicalChunker
    from docling.datamodel.base_models import ConversionStatus

    source_name = Path(file_path).name
    _parser_log(
        f"creating Docling converter for {source_name} "
        f"(ocr={'enabled' if use_ocr else 'disabled'})"
    )
    converter = _build_docling_converter(use_ocr=use_ocr)

    _parser_log(
        "starting Docling conversion; any 'Loading weights' progress bar "
        "belongs to Docling model initialization"
    )
    started_at = time.monotonic()
    try:
        result = converter.convert(source=file_path)
    except Exception as exc:
        elapsed = time.monotonic() - started_at
        _parser_log(
            f"Docling conversion failed after {elapsed:.1f}s: "
            f"{type(exc).__name__}: {exc}"
        )
        raise

    elapsed = time.monotonic() - started_at
    _parser_log(
        f"finished Docling conversion in {elapsed:.1f}s "
        f"with status={result.status.name}"
    )

    if result.status not in {
        ConversionStatus.SUCCESS,
        ConversionStatus.PARTIAL_SUCCESS,
    }:
        raise RuntimeError(
            f"Docling conversion failed for {file_path} "
            f"with status {result.status.name}"
        )
    if result.document is None:
        raise RuntimeError(f"Docling did not return a document for {file_path}")

    _parser_log("starting Docling HierarchicalChunker")
    started_at = time.monotonic()
    chunker = HierarchicalChunker()
    chunks = [
        chunk.text.strip()
        for chunk in chunker.chunk(result.document)
        if chunk.text and chunk.text.strip()
    ]
    elapsed = time.monotonic() - started_at
    _parser_log(f"finished chunking in {elapsed:.1f}s; chunks={len(chunks)}")
    return chunks


def _has_meaningful_text(markdown: str) -> bool:
    """Return true when native PDF text extraction produced usable content."""
    alnum_count = sum(1 for char in markdown if char.isalnum())
    return alnum_count >= 25


def _has_meaningful_chunks(chunks: list[str]) -> bool:
    return any(_has_meaningful_text(chunk) for chunk in chunks)


def _chunk_cache_path(doc_fingerprint: str) -> Path:
    return Path(_config.chunking.cache_dir) / f"{doc_fingerprint}.json"


def _load_chunk_cache(doc_fingerprint: str, file_path: str | None = None) -> dict | None:
    if not _config.chunking.cache_enabled:
        return None

    path = _chunk_cache_path(doc_fingerprint)
    if not path.exists():
        _parser_log(f"chunk file cache miss: {path}")
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _parser_log(f"chunk file cache ignored; failed to read {path}: {exc}")
        return None

    if payload.get("cache_version") != 1:
        _parser_log(f"chunk file cache ignored; unsupported version in {path}")
        return None

    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        _parser_log(f"chunk file cache ignored; invalid payload in {path}")
        return None

    if file_path:
        source_name = Path(file_path).name
        payload["filename"] = file_path
        payload["source_name"] = source_name
        for chunk in chunks:
            metadata = chunk.get("metadata")
            if isinstance(metadata, dict):
                metadata["source_file"] = file_path
                metadata["ref"] = source_name

    _document_chunk_cache[doc_fingerprint] = chunks
    _parser_log(f"chunk file cache hit: {path}; chunks={len(chunks)}")
    return payload


def _write_chunk_cache(payload: dict) -> None:
    if not _config.chunking.cache_enabled:
        return

    doc_fingerprint = payload["doc_fingerprint"]
    path = _chunk_cache_path(doc_fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _parser_log(f"chunk file cache saved: {path}")


def _chunk_manifest(payload: dict, cache_status: str) -> dict:
    return {
        "doc_fingerprint": payload["doc_fingerprint"],
        "filename": payload["filename"],
        "ocr_used": payload.get("ocr_used", False),
        "parse_backend": payload.get("parse_backend", "unknown"),
        "chunk_count": len(payload.get("chunks", [])),
        "chunk_cache": cache_status,
        "chunk_cache_path": str(_chunk_cache_path(payload["doc_fingerprint"])),
        "chunk_access": (
            "Full chunk text is cached locally. Extractor agents should call "
            "extract_document_triples with this doc_fingerprint; it will process "
            "cached chunks in configured batches. "
            "get_document_chunks is only for manual inspection/debugging."
        ),
    }


# ══════════════════════════════════════════════════════════════
# Parser tools
# ══════════════════════════════════════════════════════════════

@tool(description_mode="only_docstring")
def check_extraction_cache(doc_fingerprint: str) -> str:
    """
    Check if a document was already processed.

    In ontology-only Agent Memory mode this intentionally does not read
    extraction records from memory; memory is reserved for ontology entries.

    Parameters:
        doc_fingerprint (str): SHA-256 hash of the document content.

    Returns:
        str: JSON with {"already_processed": true/false}.
    """
    from oracleagentmemory.apis.searchscope import SearchScope

    _parser_log(f"check_extraction_cache started for {doc_fingerprint[:12]}")
    if _agent_memory_ontology_only():
        _parser_log(
            "check_extraction_cache skipped; Agent Memory is ontology-only"
        )
        return json.dumps({
            "already_processed": False,
            "memory_usage": "ontology_only",
        })

    results = _memory.search(
        query=f"extraction_record fingerprint {doc_fingerprint}",
        scope=SearchScope(user_id="system", agent_id="parser_agent"),
        max_results=1,
    )
    for r in (results or []):
        try:
            data = json.loads(r.content)
            if data.get("doc_fingerprint") == doc_fingerprint:
                _parser_log("check_extraction_cache completed; already_processed=True")
                return json.dumps({"already_processed": True,
                                   "record": data})
        except (json.JSONDecodeError, AttributeError):
            continue
    _parser_log("check_extraction_cache completed; already_processed=False")
    return json.dumps({"already_processed": False})


@tool(description_mode="only_docstring")
def parse_document(file_path: str) -> str:
    """
    Parse a document using Docling's HierarchicalChunker.
    PDF OCR is disabled first; OCR is only used as a fallback when Docling
    returns no meaningful chunks from the native text layer.
    Returns structured chunks with metadata.

    Parameters:
        file_path (str): Path to the document file.

    Returns:
        str: JSON document manifest. Full chunk text is cached in-process and
            extractor agents can process it with extract_document_triples using
            doc_fingerprint.
    """
    source_name = Path(file_path).name
    _parser_log(f"parse_document started for {source_name}")

    _parser_log("computing document fingerprint")
    with open(file_path, "rb") as f:
        fingerprint = hashlib.sha256(f.read()).hexdigest()

    cached_payload = _load_chunk_cache(fingerprint, file_path=file_path)
    if cached_payload is not None:
        _parser_log(
            f"parse_document completed from file cache for {source_name}; "
            f"chunks={len(cached_payload.get('chunks', []))}"
        )
        return json.dumps(
            _chunk_manifest(cached_payload, cache_status="hit"),
            ensure_ascii=False,
        )

    ocr_used = False
    parse_backend = "docling_hierarchical_no_ocr"
    chunk_texts = _chunk_document_with_docling(file_path, use_ocr=False)

    if not _has_meaningful_chunks(chunk_texts):
        _parser_log(
            "no meaningful chunks found with OCR disabled; retrying with OCR fallback"
        )
        chunk_texts = _chunk_document_with_docling(file_path, use_ocr=True)
        ocr_used = True
        parse_backend = "docling_hierarchical_ocr_fallback"
    else:
        _parser_log("meaningful chunks found; OCR fallback skipped")

    _parser_log("building chunk metadata payload for downstream agents")
    chunk_entries = []
    for chunk_idx, chunk_text in enumerate(chunk_texts):
        chunk_entries.append({
            "text": chunk_text,
            "metadata": {
                "source_file": file_path,
                "page_or_section": f"chunk_{chunk_idx}",
                "doc_fingerprint": fingerprint,
                "chunk_index": chunk_idx,
                "structural_type": "hierarchical",
                "ref": source_name,
                "uuid": str(uuid4()),
            },
        })

    _document_chunk_cache[fingerprint] = chunk_entries
    payload = {
        "cache_version": 1,
        "doc_fingerprint": fingerprint,
        "filename": file_path,
        "source_name": source_name,
        "ocr_used": ocr_used,
        "parse_backend": parse_backend,
        "chunk_count": len(chunk_entries),
        "cached_at": datetime.now().isoformat(),
        "chunks": chunk_entries,
    }
    _write_chunk_cache(payload)

    _parser_log(
        f"parse_document completed for {source_name}; "
        f"chunks={len(chunk_entries)}, backend={parse_backend}, ocr_used={ocr_used}"
    )
    return json.dumps(_chunk_manifest(payload, cache_status="saved"), ensure_ascii=False)


@tool(description_mode="only_docstring")
def get_document_chunks(
    doc_fingerprint: str,
    start: int = 0,
    limit: int = 10,
) -> str:
    """
    Retrieve cached document chunks from parse_document in small batches.

    Parameters:
        doc_fingerprint (str): SHA-256 document fingerprint returned by parse_document.
        start (int): Zero-based chunk offset.
        limit (int): Maximum number of chunks to return.

    Returns:
        str: JSON with chunks and pagination metadata.
    """
    chunks = _document_chunk_cache.get(doc_fingerprint)
    if chunks is None:
        cached_payload = _load_chunk_cache(doc_fingerprint)
        chunks = cached_payload.get("chunks", []) if cached_payload else []
    start = max(0, int(start))
    limit = max(1, min(int(limit), 20))
    end = min(start + limit, len(chunks))
    batch = chunks[start:end]
    _parser_log(
        f"get_document_chunks doc={doc_fingerprint[:12]} start={start} "
        f"limit={limit} returned={len(batch)}/{len(chunks)}"
    )
    return json.dumps({
        "doc_fingerprint": doc_fingerprint,
        "start": start,
        "limit": limit,
        "next_start": end if end < len(chunks) else None,
        "chunk_count": len(chunks),
        "chunks": batch,
    }, ensure_ascii=False)


def _get_cached_chunks_or_raise(doc_fingerprint: str) -> list[dict]:
    chunks = _document_chunk_cache.get(doc_fingerprint)
    if chunks is None:
        cached_payload = _load_chunk_cache(doc_fingerprint)
        chunks = cached_payload.get("chunks", []) if cached_payload else []
    if not chunks:
        raise RuntimeError(
            f"No cached chunks found for doc_fingerprint={doc_fingerprint}"
        )
    return chunks


# ══════════════════════════════════════════════════════════════
# Extractor tools
# ══════════════════════════════════════════════════════════════

def _stable_ontology_entries(entries: list[dict], entry_type: str) -> list[dict]:
    """Normalize, deduplicate, and sort ontology entries for stable prompts."""
    normalized: dict[object, dict] = {}
    for entry in entries:
        label = str(entry.get("label", "")).strip().lower()
        if not label:
            continue

        item = {
            "entry_type": entry_type,
            "label": label,
            "description": str(entry.get("description", "") or ""),
        }
        if entry_type == "vertex_type":
            item["naming_convention"] = str(
                entry.get("naming_convention", "") or ""
            )
        else:
            item["source"] = str(entry.get("source", "any") or "any").strip().lower()
            item["target"] = str(entry.get("target", "any") or "any").strip().lower()

        key = (
            label
            if entry_type == "vertex_type"
            else (label, item["source"], item["target"])
        )
        current = normalized.get(key)
        if current is None:
            normalized[key] = item
            continue

        # Prefer the richest deterministic entry if duplicates exist.
        current_serial = json.dumps(current, sort_keys=True)
        item_serial = json.dumps(item, sort_keys=True)
        current_score = len(current_serial)
        item_score = len(item_serial)
        if (item_score, item_serial) > (current_score, current_serial):
            normalized[key] = item

    return [
        normalized[key]
        for key in sorted(
            normalized,
            key=lambda value: value if isinstance(value, tuple) else (value, "", ""),
        )
    ]


def _domain_ontology_entries(domain_names: str) -> tuple[list[dict], list[dict], list[str]]:
    return domain_ontology_entries(
        domain_names,
        _config.ontology.domain_config_file if _config else None,
    )


@tool(description_mode="only_docstring")
def read_ontology(exclude_doc_fingerprint: str = "", domain_names: str = "core") -> str:
    """
    Read the ontology registry from Agent Memory.
    Returns allowed entity and relationship types.

    Agent Memory is intentionally used only for ontology entries. Previously
    this tool also injected existing graph canonical entity names into the
    extractor prompt, which made repeated runs depend on entities written by
    earlier runs.

    Parameters:
        exclude_doc_fingerprint (str): Optional document fingerprint whose
            own auto-admitted ontology entries should be hidden. This keeps
            repeat test runs on the same document from changing their own
            ontology context while still allowing those entries to help other
            documents.
        domain_names (str): Comma-separated ontology domain modules, for
            example "core,pharma".

    Returns:
        str: JSON with entity_types and relationship_types.
    """
    from oracleagentmemory.apis.searchscope import SearchScope

    _tool_log("extractor", "read_ontology started")
    exclude_doc_fingerprint = str(exclude_doc_fingerprint or "").strip()
    domain_vertices, domain_edges, active_domains = _domain_ontology_entries(
        domain_names
    )
    results = _memory.search(
        query="ontology_entry vertex_type relationship_type",
        scope=SearchScope(user_id="system", agent_id="ontology_manager"),
        max_results=500,
    )
    entity_candidates = list(domain_vertices)
    relationship_candidates = list(domain_edges)
    for r in (results or []):
        try:
            data = json.loads(r.content)
            if (
                exclude_doc_fingerprint
                and data.get("source_doc_fingerprint") == exclude_doc_fingerprint
            ):
                continue
            if data.get("entry_type") == "vertex_type":
                entity_candidates.append(data)
            elif data.get("entry_type") == "edge_type":
                relationship_candidates.append(data)
        except (json.JSONDecodeError, AttributeError):
            continue

    entity_types = _stable_ontology_entries(entity_candidates, "vertex_type")
    relationship_types = _stable_ontology_entries(
        relationship_candidates,
        "edge_type",
    )

    _tool_log(
        "extractor",
        "read_ontology completed; "
        f"entity_types={len(entity_types)}, "
        f"relationship_types={len(relationship_types)}"
    )
    return json.dumps({
        "entity_types": entity_types,
        "relationship_types": relationship_types,
        "known_canonicals": [],
        "memory_usage": "ontology_only",
        "excluded_doc_fingerprint": exclude_doc_fingerprint,
        "active_domains": active_domains,
    }, ensure_ascii=False)


# ── Extraction prompt templates ──────────────────────────────

EXTRACTION_PROMPTS = {
    "entity_first": """You are a knowledge graph extraction expert.

TASK: Identify all entities in the text, classify them by type, then
describe the relationships between them.

ALLOWED ENTITY TYPES: {entity_types}
ALLOWED RELATIONSHIP TYPES: {relationship_types}
ALLOWED RELATIONSHIP SIGNATURES: {relationship_signatures}

TEXT:
{chunk_text}

Rules:
- Triples MUST use only the allowed entity and relationship types listed above.
- Triple subject/object types MUST match one allowed relationship signature for the selected relationship. The source and target value "any" is a wildcard.
- Do not invent entity types or relationship types inside triples.
- If a useful reusable type is missing, omit that unsupported triple and add an ontology_candidates item instead.
- Candidate labels must be generic reusable types, not specific entity names, dates, counts, measurements, or one-off values.
- Every triple must be directly grounded in this text.

Respond ONLY with a JSON object:
{{
  "triples": [
    {{
      "subject": {{"name": "...", "type": "...", "properties": {{}}}},
      "relationship": "...",
      "object": {{"name": "...", "type": "...", "properties": {{}}}},
      "confidence": 0.0-1.0
    }}
  ],
  "ontology_candidates": [
    {{
      "kind": "vertex_type",
      "label": "reusable_snake_case_type",
      "description": "why this reusable type is needed",
      "evidence": "short quote or phrase from the text",
      "source_chunk_index": 0
    }}
  ],
  "pending_types": []
}}
""",

    "relationship_first": """You are a knowledge graph extraction expert.

TASK: Find all relationships described in the text, then identify
the entities involved in each relationship.

ALLOWED ENTITY TYPES: {entity_types}
ALLOWED RELATIONSHIP TYPES: {relationship_types}
ALLOWED RELATIONSHIP SIGNATURES: {relationship_signatures}

TEXT:
{chunk_text}

Rules:
- Triples MUST use only the allowed entity and relationship types listed above.
- Triple subject/object types MUST match one allowed relationship signature for the selected relationship. The source and target value "any" is a wildcard.
- Do not invent entity types or relationship types inside triples.
- If a useful reusable type is missing, omit that unsupported triple and add an ontology_candidates item instead.
- Candidate labels must be generic reusable types, not specific entity names, dates, counts, measurements, or one-off values.
- Every triple must be directly grounded in this text.

Respond ONLY with a JSON object:
{{
  "triples": [
    {{
      "subject": {{"name": "...", "type": "...", "properties": {{}}}},
      "relationship": "...",
      "object": {{"name": "...", "type": "...", "properties": {{}}}},
      "confidence": 0.0-1.0
    }}
  ],
  "ontology_candidates": [
    {{
      "kind": "edge_type",
      "label": "reusable_snake_case_relationship",
      "source": "allowed_source_type_or_any",
      "target": "allowed_target_type_or_any",
      "description": "why this reusable relationship type is needed",
      "evidence": "short quote or phrase from the text",
      "source_chunk_index": 0
    }}
  ],
  "pending_types": []
}}
""",

    "balanced": """You are a knowledge graph extraction expert.

TASK: Extract entities and their relationships from the text.

ALLOWED ENTITY TYPES: {entity_types}
ALLOWED RELATIONSHIP TYPES: {relationship_types}
ALLOWED RELATIONSHIP SIGNATURES: {relationship_signatures}

TEXT:
{chunk_text}

Rules:
- Triples MUST use only the allowed entity and relationship types listed above.
- Triple subject/object types MUST match one allowed relationship signature for the selected relationship. The source and target value "any" is a wildcard.
- Do not invent entity types or relationship types inside triples.
- If a useful reusable type is missing, omit that unsupported triple and add an ontology_candidates item instead.
- Candidate labels must be generic reusable types, not specific entity names, dates, counts, measurements, or one-off values.
- Every triple must be directly grounded in this text.

Respond ONLY with a JSON object:
{{
  "triples": [
    {{
      "subject": {{"name": "...", "type": "...", "properties": {{}}}},
      "relationship": "...",
      "object": {{"name": "...", "type": "...", "properties": {{}}}},
      "confidence": 0.0-1.0
    }}
  ],
  "ontology_candidates": [
    {{
      "kind": "vertex_type",
      "label": "reusable_snake_case_type",
      "description": "why this reusable type is needed",
      "evidence": "short quote or phrase from the text",
      "source_chunk_index": 0
    }}
  ],
  "pending_types": []
}}
""",
}


def _ontology_prompt_values(ontology_json: str) -> tuple[str, str, str]:
    ontology = json.loads(ontology_json)
    entity_type_names = sorted({
        str(et.get("label", "")).strip().lower()
        for et in ontology.get("entity_types", [])
        if isinstance(et, dict) and str(et.get("label", "")).strip()
    })
    rel_type_names = sorted({
        str(rt.get("label", "")).strip().lower()
        for rt in ontology.get("relationship_types", [])
        if isinstance(rt, dict) and str(rt.get("label", "")).strip()
    })
    rel_signatures = sorted({
        (
            _normalize_extracted_label(rt.get("label", "")),
            _normalize_signature_endpoint(rt.get("source", "any")),
            _normalize_signature_endpoint(rt.get("target", "any")),
        )
        for rt in ontology.get("relationship_types", [])
        if isinstance(rt, dict) and str(rt.get("label", "")).strip()
    })
    rel_signature_text = [
        f"{label}: {source} -> {target}"
        for label, source, target in rel_signatures
        if label
    ]
    return (
        ", ".join(entity_type_names) or "any",
        ", ".join(rel_type_names) or "any",
        "; ".join(rel_signature_text) or "any -> any",
    )


def _ontology_allowed_sets(
    ontology_json: str,
) -> tuple[set[str], set[str], dict[str, set[tuple[str, str]]]]:
    ontology = json.loads(ontology_json)
    entity_types = {
        _normalize_extracted_label(et.get("label", ""))
        for et in ontology.get("entity_types", [])
        if isinstance(et, dict) and str(et.get("label", "")).strip()
    }
    relationship_types = set()
    relationship_signatures: dict[str, set[tuple[str, str]]] = {}
    for rt in ontology.get("relationship_types", []):
        if not isinstance(rt, dict) or not str(rt.get("label", "")).strip():
            continue
        label = _normalize_extracted_label(rt.get("label", ""))
        if not label:
            continue
        relationship_types.add(label)
        relationship_signatures.setdefault(label, set()).add((
            _normalize_signature_endpoint(rt.get("source", "any")),
            _normalize_signature_endpoint(rt.get("target", "any")),
        ))
    entity_types.discard("")
    relationship_types.discard("")
    return entity_types, relationship_types, relationship_signatures


def _normalize_extracted_label(value: object) -> str:
    label = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return label.strip("_")


def _normalize_signature_endpoint(value: object) -> str:
    label = _normalize_extracted_label(value)
    return "any" if label in {"", "any", "*"} else label


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


def _strict_ontology_extraction() -> bool:
    ensemble = getattr(_config, "ensemble", None)
    return bool(getattr(ensemble, "strict_ontology", True))


def _entity_name_from_payload(entity: object) -> str:
    if not isinstance(entity, dict):
        return ""
    return " ".join(str(entity.get("name", "")).strip().split())


def _normalize_properties(entity: object) -> dict:
    if not isinstance(entity, dict):
        return {}
    properties = entity.get("properties", {})
    return properties if isinstance(properties, dict) else {}


def _normalize_ontology_candidates(
    data: dict,
    allowed_entity_types: set[str],
    allowed_relationship_types: set[str],
    default_chunk_index: int | None = None,
) -> list[dict]:
    candidates = data.get("ontology_candidates", [])
    if not isinstance(candidates, list):
        candidates = []

    normalized: list[dict] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        kind = _normalize_extracted_label(
            candidate.get("kind")
            or candidate.get("entry_type")
            or candidate.get("type_kind")
        )
        label = _normalize_extracted_label(candidate.get("label") or candidate.get("name"))
        if kind in {"entity", "entity_type", "vertex"}:
            kind = "vertex_type"
        elif kind in {"relationship", "relationship_type", "edge"}:
            kind = "edge_type"
        if kind not in {"vertex_type", "edge_type"} or not label:
            continue
        if kind == "vertex_type" and label in allowed_entity_types:
            continue
        if kind == "edge_type" and label in allowed_relationship_types:
            continue

        try:
            source_chunk_index = int(
                candidate.get("source_chunk_index", default_chunk_index)
            )
        except (TypeError, ValueError):
            source_chunk_index = default_chunk_index

        item = {
            "kind": kind,
            "label": label,
            "description": str(candidate.get("description", "")).strip(),
            "evidence": str(candidate.get("evidence", "")).strip(),
            "source_chunk_index": source_chunk_index,
        }
        if kind == "edge_type":
            source = _normalize_extracted_label(candidate.get("source") or "any")
            target = _normalize_extracted_label(candidate.get("target") or "any")
            item["source"] = source if source in allowed_entity_types else "any"
            item["target"] = target if target in allowed_entity_types else "any"
        normalized.append(item)

    return normalized


def _legacy_pending_types_to_candidates(
    pending_types: object,
    default_chunk_index: int | None = None,
) -> list[dict]:
    if not isinstance(pending_types, list):
        return []
    candidates: list[dict] = []
    for pending_type in pending_types:
        if isinstance(pending_type, dict):
            kind = _normalize_extracted_label(
                pending_type.get("kind")
                or pending_type.get("entry_type")
                or pending_type.get("type_kind")
                or ""
            )
            label = _normalize_extracted_label(
                pending_type.get("label") or pending_type.get("name") or ""
            )
        else:
            raw = str(pending_type or "").strip()
            if ":" in raw:
                kind_raw, label_raw = raw.split(":", 1)
                kind = _normalize_extracted_label(kind_raw)
                label = _normalize_extracted_label(label_raw)
            else:
                kind = ""
                label = _normalize_extracted_label(raw)

        if kind in {"entity", "entity_type", "vertex"}:
            kind = "vertex_type"
        elif kind in {"relationship", "relationship_type", "edge"}:
            kind = "edge_type"
        if kind not in {"vertex_type", "edge_type"} or not label:
            continue
        candidates.append({
            "kind": kind,
            "label": label,
            "description": "Legacy pending type emitted by extractor.",
            "evidence": "",
            "source_chunk_index": default_chunk_index,
        })
    return candidates


def _normalize_extracted_triples(
    triples: object,
    chunk_lookup: dict[int, str],
    allowed_entity_types: set[str],
    allowed_relationship_types: set[str],
    relationship_signatures: dict[str, set[tuple[str, str]]],
) -> tuple[list[dict], int, dict[int, int]]:
    if not isinstance(triples, list):
        triples = []

    normalized_triples = []
    skipped = 0
    per_chunk_counts = {chunk_index: 0 for chunk_index in chunk_lookup}
    strict = _strict_ontology_extraction()

    for triple in triples:
        if not isinstance(triple, dict):
            skipped += 1
            continue
        raw_chunk_index = triple.get("source_chunk_index", triple.get("chunk_index"))
        try:
            chunk_index = int(raw_chunk_index)
        except (TypeError, ValueError):
            if len(chunk_lookup) == 1:
                chunk_index = next(iter(chunk_lookup))
            else:
                skipped += 1
                continue
        if chunk_index not in chunk_lookup:
            skipped += 1
            continue

        subject = triple.get("subject")
        obj = triple.get("object")
        subject_name = _entity_name_from_payload(subject)
        object_name = _entity_name_from_payload(obj)
        subject_type = _normalize_extracted_label(
            subject.get("type") if isinstance(subject, dict) else ""
        )
        object_type = _normalize_extracted_label(
            obj.get("type") if isinstance(obj, dict) else ""
        )
        relationship = _normalize_extracted_label(triple.get("relationship"))
        if not subject_name or not object_name or not relationship:
            skipped += 1
            continue

        if strict and (
            subject_type not in allowed_entity_types
            or object_type not in allowed_entity_types
            or relationship not in allowed_relationship_types
            or not _relationship_signature_allowed(
                relationship,
                subject_type,
                object_type,
                relationship_signatures,
            )
        ):
            skipped += 1
            continue

        try:
            confidence = float(triple.get("confidence", 0.8) or 0.8)
        except (TypeError, ValueError):
            confidence = 0.8
        confidence = max(0.0, min(1.0, confidence))

        normalized_triples.append({
            "source_chunk_index": chunk_index,
            "source_doc": chunk_lookup[chunk_index],
            "subject": {
                "name": subject_name,
                "type": subject_type,
                "properties": _normalize_properties(subject),
            },
            "relationship": relationship,
            "object": {
                "name": object_name,
                "type": object_type,
                "properties": _normalize_properties(obj),
            },
            "confidence": confidence,
        })
        per_chunk_counts[chunk_index] += 1

    return normalized_triples, skipped, per_chunk_counts


def _batch_strategy_text(prompt_angle: str) -> str:
    if prompt_angle == "entity_first":
        return (
            "Identify all entities in each chunk first, classify them by type, "
            "then describe relationships between entities from the same chunk."
        )
    if prompt_angle == "relationship_first":
        return (
            "Find all relationships described in each chunk first, then identify "
            "the entities involved in each relationship."
        )
    return "Extract entities and their relationships from each chunk."


def _extract_triples_data(
    chunk_text: str,
    chunk_index: int,
    source_doc: str,
    model: str,
    temperature: float,
    prompt_angle: str,
    ontology_json: str,
) -> dict:
    _tool_log(
        "extractor",
        f"extract_triples started; chunk={chunk_index}, "
        f"model={model}, angle={prompt_angle}, chars={len(chunk_text)}"
    )
    entity_types, relationship_types, relationship_signatures_text = _ontology_prompt_values(
        ontology_json
    )
    (
        allowed_entity_types,
        allowed_relationship_types,
        relationship_signatures,
    ) = _ontology_allowed_sets(
        ontology_json
    )

    prompt_template = EXTRACTION_PROMPTS.get(prompt_angle, EXTRACTION_PROMPTS["balanced"])
    prompt = prompt_template.format(
        entity_types=entity_types,
        relationship_types=relationship_types,
        relationship_signatures=relationship_signatures_text,
        chunk_text=chunk_text,
    )

    _tool_log(
        "extractor",
        f"calling OpenAI for chunk={chunk_index}, model={model}"
    )
    started_at = time.monotonic()
    response = _openai.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    elapsed = time.monotonic() - started_at

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"triples": [], "pending_types": [], "ontology_candidates": []}

    chunk_lookup = {chunk_index: source_doc}
    triples, skipped, per_chunk_counts = _normalize_extracted_triples(
        data.get("triples", []),
        chunk_lookup,
        allowed_entity_types,
        allowed_relationship_types,
        relationship_signatures,
    )
    ontology_candidates = _normalize_ontology_candidates(
        data,
        allowed_entity_types,
        allowed_relationship_types,
        default_chunk_index=chunk_index,
    )
    ontology_candidates.extend(
        _legacy_pending_types_to_candidates(
            data.get("pending_types", []),
            default_chunk_index=chunk_index,
        )
    )
    data = {
        "triples": triples,
        "pending_types": data.get("pending_types", []),
        "ontology_candidates": ontology_candidates,
        "chunk_counts": [
            {"chunk_index": idx, "triple_count": count}
            for idx, count in per_chunk_counts.items()
        ],
        "skipped_triples": skipped,
    }

    _tool_log(
        "extractor",
        f"extract_triples completed; chunk={chunk_index}, "
        f"triples={len(data.get('triples', []))}, elapsed={elapsed:.1f}s"
    )
    return data


def _extract_triples_batch_data(
    batch_chunks: list[dict],
    batch_number: int,
    total_batches: int,
    model: str,
    temperature: float,
    prompt_angle: str,
    ontology_json: str,
    source_doc_override: str = "",
) -> dict:
    entity_types, relationship_types, relationship_signatures_text = _ontology_prompt_values(
        ontology_json
    )
    (
        allowed_entity_types,
        allowed_relationship_types,
        relationship_signatures,
    ) = _ontology_allowed_sets(
        ontology_json
    )

    chunk_lookup: dict[int, str] = {}
    chunk_payload = []
    total_chars = 0
    for chunk in batch_chunks:
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        chunk_index = int(metadata.get("chunk_index", len(chunk_payload)))
        source_doc = (
            source_doc_override
            or metadata.get("source_file")
            or metadata.get("ref")
            or ""
        )
        chunk_lookup[chunk_index] = source_doc
        chunk_payload.append({
            "chunk_index": chunk_index,
            "source_doc": source_doc,
            "text": text,
        })
        total_chars += len(text)

    prompt = f"""You are a knowledge graph extraction expert.

TASK: {_batch_strategy_text(prompt_angle)}

ALLOWED ENTITY TYPES: {entity_types}
ALLOWED RELATIONSHIP TYPES: {relationship_types}
ALLOWED RELATIONSHIP SIGNATURES: {relationship_signatures_text}

INPUT CHUNKS JSON:
{json.dumps(chunk_payload, ensure_ascii=False, indent=2)}

Rules:
- Extract only facts directly grounded in the input chunks.
- Do not combine facts across different chunks.
- Every triple MUST include source_chunk_index, matching one input chunk_index.
- Triples MUST use only the allowed entity and relationship types listed above.
- Triple subject/object types MUST match one allowed relationship signature for the selected relationship. The source and target value "any" is a wildcard.
- Do not invent entity types or relationship types inside triples.
- If a fact requires a missing type, omit that unsupported triple and add an ontology_candidates item instead.
- Candidate labels must be generic reusable schema types, not specific entity names, dates, counts, measurements, or one-off values.
- Prefer precise canonical entity names from the text, but keep them stable and complete.

Respond ONLY with a JSON object:
{{
  "triples": [
    {{
      "source_chunk_index": 0,
      "subject": {{"name": "...", "type": "...", "properties": {{}}}},
      "relationship": "...",
      "object": {{"name": "...", "type": "...", "properties": {{}}}},
      "confidence": 0.0
    }}
  ],
  "ontology_candidates": [
    {{
      "kind": "vertex_type",
      "label": "reusable_snake_case_type",
      "description": "why this reusable type is needed",
      "evidence": "short quote or phrase from the input chunk",
      "source_chunk_index": 0
    }},
    {{
      "kind": "edge_type",
      "label": "reusable_snake_case_relationship",
      "source": "allowed_source_type_or_any",
      "target": "allowed_target_type_or_any",
      "description": "why this reusable relationship type is needed",
      "evidence": "short quote or phrase from the input chunk",
      "source_chunk_index": 0
    }}
  ],
  "pending_types": []
}}
"""

    first_index = chunk_payload[0]["chunk_index"] if chunk_payload else None
    last_index = chunk_payload[-1]["chunk_index"] if chunk_payload else None
    _tool_log(
        "extractor",
        f"calling OpenAI for batch={batch_number}/{total_batches}, "
        f"chunks={len(batch_chunks)}, chunk_index_range={first_index}-{last_index}, "
        f"model={model}, chars={total_chars}"
    )
    started_at = time.monotonic()
    response = _openai.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    elapsed = time.monotonic() - started_at

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"triples": [], "pending_types": [], "ontology_candidates": []}

    normalized_triples, skipped, per_chunk_counts = _normalize_extracted_triples(
        data.get("triples", []),
        chunk_lookup,
        allowed_entity_types,
        allowed_relationship_types,
        relationship_signatures,
    )
    ontology_candidates = _normalize_ontology_candidates(
        data,
        allowed_entity_types,
        allowed_relationship_types,
    )
    ontology_candidates.extend(
        _legacy_pending_types_to_candidates(data.get("pending_types", []))
    )

    pending_types = data.get("pending_types", [])
    if not isinstance(pending_types, list):
        pending_types = []

    _tool_log(
        "extractor",
        f"batch completed; batch={batch_number}/{total_batches}, "
        f"triples={len(normalized_triples)}, skipped={skipped}, "
        f"elapsed={elapsed:.1f}s"
    )
    return {
        "triples": normalized_triples,
        "pending_types": pending_types,
        "ontology_candidates": ontology_candidates,
        "chunk_counts": [
            {"chunk_index": chunk_index, "triple_count": count}
            for chunk_index, count in per_chunk_counts.items()
        ],
        "skipped_triples": skipped,
    }


@tool(description_mode="only_docstring")
def extract_triples(
    chunk_text: str,
    chunk_index: int,
    source_doc: str,
    model: str,
    temperature: float,
    prompt_angle: str,
    ontology_json: str,
) -> str:
    """
    Extract entity-relationship triples from a text chunk using the
    specified model, temperature, and prompt strategy.

    Parameters:
        chunk_text (str): The text chunk to extract from.
        chunk_index (int): Index of the chunk in the document.
        source_doc (str): Source document filename.
        model (str): OpenAI model name (e.g. "gpt-4o").
        temperature (float): Sampling temperature.
        prompt_angle (str): One of "entity_first", "relationship_first", "balanced".
        ontology_json (str): JSON string with entity_types and relationship_types.

    Returns:
        str: JSON with extracted triples.
    """
    data = _extract_triples_data(
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        source_doc=source_doc,
        model=model,
        temperature=temperature,
        prompt_angle=prompt_angle,
        ontology_json=ontology_json,
    )
    return json.dumps(data, ensure_ascii=False)


@tool(description_mode="only_docstring")
def extract_document_triples(
    doc_fingerprint: str,
    model: str,
    temperature: float,
    prompt_angle: str,
    ontology_json: str,
    source_doc: str = "",
) -> str:
    """
    Extract triples from every cached chunk of a parsed document.
    This tool handles batched chunk iteration internally so agents do not need to
    paginate with get_document_chunks.

    Parameters:
        doc_fingerprint (str): SHA-256 document fingerprint from parse_document.
        model (str): OpenAI model name.
        temperature (float): Sampling temperature.
        prompt_angle (str): One of "entity_first", "relationship_first", "balanced".
        ontology_json (str): JSON string with entity_types and relationship_types.
        source_doc (str): Optional source document filename.

    Returns:
        str: JSON with all extracted triples, batch count, and per-chunk counts.
    """
    chunks = _get_cached_chunks_or_raise(doc_fingerprint)
    total = len(chunks)
    batch_size = max(1, int(_config.ensemble.extraction_batch_size))
    total_batches = (total + batch_size - 1) // batch_size
    all_triples = []
    pending_types = set()
    ontology_candidates = []
    chunk_counts = []
    skipped_triples = 0

    _tool_log(
        "extractor",
        f"extract_document_triples started; doc={doc_fingerprint[:12]}, "
        f"chunks={total}, batch_size={batch_size}, batches={total_batches}, "
        f"model={model}, angle={prompt_angle}"
    )
    started_at = time.monotonic()

    for batch_number, start in enumerate(range(0, total, batch_size), start=1):
        batch_chunks = chunks[start:start + batch_size]
        end = start + len(batch_chunks)
        _tool_log(
            "extractor",
            f"extract_document_triples progress; batch={batch_number}/{total_batches}, "
            f"chunks={start + 1}-{end}/{total}"
        )
        data = _extract_triples_batch_data(
            batch_chunks=batch_chunks,
            batch_number=batch_number,
            total_batches=total_batches,
            model=model,
            temperature=temperature,
            prompt_angle=prompt_angle,
            ontology_json=ontology_json,
            source_doc_override=source_doc,
        )
        triples = data.get("triples", [])
        all_triples.extend(triples)
        for pending_type in data.get("pending_types", []):
            if pending_type:
                pending_type_name = str(pending_type).strip()
                if pending_type_name:
                    pending_types.add(pending_type_name)
        chunk_counts.extend(data.get("chunk_counts", []))
        for candidate in data.get("ontology_candidates", []):
            if isinstance(candidate, dict):
                ontology_candidates.append(candidate)
        skipped_triples += int(data.get("skipped_triples", 0))

    elapsed = time.monotonic() - started_at
    _tool_log(
        "extractor",
        f"extract_document_triples completed; doc={doc_fingerprint[:12]}, "
        f"chunks={total}, batches={total_batches}, triples={len(all_triples)}, "
        f"skipped={skipped_triples}, elapsed={elapsed:.1f}s"
    )
    return json.dumps({
        "doc_fingerprint": doc_fingerprint,
        "model": model,
        "prompt_angle": prompt_angle,
        "extraction_batch_size": batch_size,
        "batches_processed": total_batches,
        "chunks_processed": total,
        "chunk_counts": chunk_counts,
        "triples": all_triples,
        "ontology_candidates": ontology_candidates,
        "pending_types": sorted(pending_types),
        "skipped_triples": skipped_triples,
    }, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════
# Consensus Merger tools
# ══════════════════════════════════════════════════════════════

def _normalize_name(name: str) -> str:
    """Normalize entity name for comparison."""
    name = unicodedata.normalize("NFC", name)
    name = name.strip().lower()
    name = " ".join(name.split())
    return name


@tool(description_mode="only_docstring")
def compute_similarity(name_a: str, name_b: str) -> str:
    """
    Compute cosine similarity between two entity names using
    the configured OpenAI embedding model.

    Parameters:
        name_a (str): First entity name.
        name_b (str): Second entity name.

    Returns:
        str: JSON with {"similarity": float}.
    """
    _tool_log("consensus", f"compute_similarity started: '{name_a}' vs '{name_b}'")
    embeddings = _embed_batch([name_a, name_b])
    # Cosine similarity
    import numpy as np
    a = np.array(embeddings[0])
    b = np.array(embeddings[1])
    sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    _tool_log("consensus", f"compute_similarity completed; similarity={sim:.4f}")
    return json.dumps({"similarity": round(sim, 4)})


@tool(description_mode="only_docstring")
def log_consensus_result(result_json: str) -> str:
    """
    Log a consensus decision.

    In ontology-only Agent Memory mode this is a no-op; memory is reserved
    for ontology entries.

    Parameters:
        result_json (str): JSON with triple, agreeing/disagreeing extractors, status.

    Returns:
        str: Confirmation JSON.
    """
    _tool_log("consensus", "log_consensus_result started")
    if _agent_memory_ontology_only():
        _tool_log("consensus", "log_consensus_result skipped; ontology-only memory")
        return json.dumps({"logged": False, "memory_usage": "ontology_only"})

    _memory.add_memory(
        result_json,
        user_id="system",
        agent_id="consensus_merger",
    )
    _tool_log("consensus", "log_consensus_result completed")
    return json.dumps({"logged": True})


# ══════════════════════════════════════════════════════════════
# Reconciler tools
# ══════════════════════════════════════════════════════════════

@tool(description_mode="only_docstring")
def search_decision_log(entity_name: str) -> str:
    """
    Search for prior merge/canonicalization decisions
    about a given entity name.

    In ontology-only Agent Memory mode this always returns no decisions.

    Parameters:
        entity_name (str): The entity surface form to look up.

    Returns:
        str: JSON with prior decisions if found, empty list otherwise.
    """
    from oracleagentmemory.apis.searchscope import SearchScope

    _tool_log("reconciler", f"search_decision_log started; entity={entity_name}")
    if _agent_memory_ontology_only():
        _tool_log("reconciler", "search_decision_log skipped; ontology-only memory")
        return json.dumps({
            "decisions": [],
            "memory_usage": "ontology_only",
        }, ensure_ascii=False)

    results = _memory.search(
        query=f"merge_decision {entity_name}",
        scope=SearchScope(user_id="system", agent_id="reconciler_agent"),
        max_results=5,
    )
    decisions = []
    for r in (results or []):
        try:
            data = json.loads(r.content)
            if data.get("memory_type") == "merge_decision":
                decisions.append(data)
        except (json.JSONDecodeError, AttributeError):
            continue
    _tool_log(
        "reconciler",
        f"search_decision_log completed; entity={entity_name}, "
        f"decisions={len(decisions)}"
    )
    return json.dumps({"decisions": decisions}, ensure_ascii=False)


@tool(description_mode="only_docstring")
def save_decision(decision_json: str) -> str:
    """
    Save a merge/canonicalization decision.

    In ontology-only Agent Memory mode this is a no-op.

    Parameters:
        decision_json (str): JSON with surface_form, canonical_form,
            entity_type, action, confidence, reasoning.

    Returns:
        str: Confirmation JSON.
    """
    _tool_log("reconciler", "save_decision started")
    if _agent_memory_ontology_only():
        _tool_log("reconciler", "save_decision skipped; ontology-only memory")
        return json.dumps({"saved": False, "memory_usage": "ontology_only"})

    _memory.add_memory(
        decision_json,
        user_id="system",
        agent_id="reconciler_agent",
    )
    _tool_log("reconciler", "save_decision completed")
    return json.dumps({"saved": True})


@tool(description_mode="only_docstring")
def lookup_similar_vertices(entity_name: str, entity_type: str) -> str:
    """
    Search the Property Graph vertex table for existing vertices
    similar to the given entity name, using vector similarity.

    Parameters:
        entity_name (str): The entity name to search for.
        entity_type (str): The entity type to filter by.

    Returns:
        str: JSON list of similar vertices with similarity scores.
    """
    _tool_log(
        "reconciler",
        f"lookup_similar_vertices started; entity={entity_name}, type={entity_type}"
    )
    embedding = _embed(entity_name)

    with _pool.acquire() as conn:
        with conn.cursor() as cur:
            _set_clob_inputs(cur, "query_embedding")
            try:
                cur.execute(f"""
                    SELECT vertex_id, canonical_name, vertex_type,
                           VECTOR_DISTANCE(
                               name_embedding,
                               TO_VECTOR(:query_embedding, {_config.openai.embedding_dimensions}, FLOAT32),
                               COSINE
                           ) AS distance
                    FROM {_config.vertex_table}
                    WHERE vertex_type = :entity_type
                    ORDER BY VECTOR_DISTANCE(
                        name_embedding,
                        TO_VECTOR(:query_embedding, {_config.openai.embedding_dimensions}, FLOAT32),
                        COSINE
                    )
                    FETCH FIRST 5 ROWS ONLY
                """, {
                    "query_embedding": _vector_literal(embedding),
                    "entity_type": entity_type,
                })
            except oracledb.DatabaseError as exc:
                if _is_missing_object_error(exc):
                    _tool_log(
                        "reconciler",
                        "lookup_similar_vertices skipped; vertex table missing"
                    )
                    return json.dumps({"similar_vertices": []}, ensure_ascii=False)
                raise
            results = []
            for row in cur.fetchall():
                similarity = 1.0 - (row[3] or 0.0)
                if similarity >= _config.reconciler.similarity_threshold:
                    results.append({
                        "vertex_id": row[0],
                        "canonical_name": row[1],
                        "vertex_type": row[2],
                        "similarity": round(similarity, 4),
                    })

    _tool_log(
        "reconciler",
        f"lookup_similar_vertices completed; entity={entity_name}, "
        f"matches={len(results)}"
    )
    return json.dumps({"similar_vertices": results}, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════
# GraphWriter tools
# ══════════════════════════════════════════════════════════════

@tool(description_mode="only_docstring")
def create_graph_ddl(ontology_json: str) -> str:
    """
    Generate and execute CREATE PROPERTY GRAPH DDL from the ontology.
    Called on the first run only.

    Parameters:
        ontology_json (str): JSON with entity_types and relationship_types.

    Returns:
        str: Confirmation or error.
    """
    from schema import (
        generate_index_ddls, generate_table_ddls,
        generate_property_graph_ddl,
        generate_typed_graph_materialized_view_ddls,
        typed_graph_materialized_view_names,
    )

    ontology = json.loads(ontology_json)
    vertex_types = ontology.get("entity_types", [])
    edge_types = ontology.get("relationship_types", [])

    _tool_log(
        "graphwriter",
        f"create_graph_ddl started; vertex_types={len(vertex_types)}, "
        f"edge_types={len(edge_types)}"
    )
    with _pool.acquire() as conn:
        with conn.cursor() as cur:
            # Create base tables
            for ddl in generate_table_ddls(_config):
                try:
                    cur.execute(ddl)
                except oracledb.DatabaseError as e:
                    if not _is_already_exists_error(e):
                        raise

            # Create indexes
            for stmt in generate_index_ddls(_config):
                stmt = stmt.strip()
                if stmt:
                    try:
                        cur.execute(stmt)
                    except oracledb.DatabaseError:
                        pass  # index may already exist

            vertex_types, edge_types = _cumulative_ontology_entries(
                cur,
                vertex_types,
                edge_types,
            )
            _tool_log(
                "graphwriter",
                "cumulative graph ontology prepared; "
                f"vertex_types={len(vertex_types)}, edge_types={len(edge_types)}"
            )

            # Create typed materialized views used by the property graph labels.
            typed_mv_ddls = generate_typed_graph_materialized_view_ddls(
                _config, vertex_types, edge_types
            )
            typed_mv_names = typed_graph_materialized_view_names(
                _config, vertex_types, edge_types
            )
            _tool_log(
                "graphwriter",
                f"typed graph objects prepared; materialized_views={len(typed_mv_names)}"
            )
            _create_materialized_views(cur, typed_mv_ddls)

            # Create property graph
            pg_ddl = generate_property_graph_ddl(
                _config, vertex_types, edge_types
            )
            _tool_log(
                "graphwriter",
                f"property graph DDL prepared; chars={len(pg_ddl)}"
            )
            try:
                cur.execute(pg_ddl)
            except oracledb.DatabaseError as e:
                if not _is_already_exists_error(e):
                    raise
                if not _property_graph_has_typed_labels(
                    cur, _config.graph_name, vertex_types, edge_types
                ):
                    _tool_log(
                        "graphwriter",
                        "property graph exists without typed labels; "
                        "recreating graph metadata"
                    )
                    cur.execute(f"DROP PROPERTY GRAPH {_config.graph_name}")
                    _drop_materialized_views(cur, typed_mv_names)
                    _create_materialized_views(cur, typed_mv_ddls)
                    cur.execute(pg_ddl)

            conn.commit()

    _tool_log("graphwriter", f"create_graph_ddl completed; graph={_config.graph_name}")
    return json.dumps({"created": True, "graph_name": _config.graph_name})


@tool(description_mode="only_docstring")
def refresh_typed_graph_views(ontology_json: str) -> str:
    """
    Complete-refresh typed graph materialized views after vertex/edge writes.

    Parameters:
        ontology_json (str): JSON with entity_types and relationship_types.

    Returns:
        str: JSON confirmation with refreshed materialized view count.
    """
    from schema import typed_graph_materialized_view_names

    ontology = json.loads(ontology_json)
    vertex_types = ontology.get("entity_types", [])
    edge_types = ontology.get("relationship_types", [])

    refreshed = 0
    with _pool.acquire() as conn:
        with conn.cursor() as cur:
            vertex_types, edge_types = _cumulative_ontology_entries(
                cur,
                vertex_types,
                edge_types,
            )
            names = typed_graph_materialized_view_names(
                _config,
                vertex_types,
                edge_types,
            )
            _tool_log(
                "graphwriter",
                f"refresh_typed_graph_views started; views={len(names)}"
            )
            for name in names:
                try:
                    cur.execute(
                        "BEGIN DBMS_MVIEW.REFRESH(:name, 'C'); END;",
                        {"name": name},
                    )
                    refreshed += 1
                    if refreshed % 50 == 0:
                        _tool_log(
                            "graphwriter",
                            f"refresh_typed_graph_views progress; refreshed={refreshed}/{len(names)}"
                        )
                except oracledb.DatabaseError as exc:
                    if not _is_missing_object_error(exc):
                        raise
            conn.commit()

    _tool_log(
        "graphwriter",
        f"refresh_typed_graph_views completed; refreshed={refreshed}"
    )
    return json.dumps({"refreshed": refreshed})


@tool(description_mode="only_docstring")
def merge_vertex(
    vertex_id: str,
    canonical_name: str,
    vertex_type: str,
    properties_json: str,
    doc_fingerprint: str,
    source_doc: str,
    source_chunk: int,
    confidence: float,
    consensus_count: int,
) -> str:
    """
    Idempotent upsert of a vertex into the vertex table.
    Generates embedding for the canonical name.

    Parameters:
        vertex_id (str): Unique vertex identifier.
        canonical_name (str): Canonical entity name.
        vertex_type (str): Entity type label.
        properties_json (str): JSON properties string.
        doc_fingerprint (str): Source document fingerprint.
        source_doc (str): Source document filename.
        source_chunk (int): Source chunk index.
        confidence (float): Extraction confidence.
        consensus_count (int): Number of agreeing extractors.

    Returns:
        str: JSON confirmation with action taken.
    """
    from schema import generate_merge_vertex_doc_sql, generate_merge_vertex_sql

    _tool_log(
        "graphwriter",
        f"merge_vertex started; id={vertex_id}, type={vertex_type}"
    )
    embedding = _embed(canonical_name)

    with _pool.acquire() as conn:
        with conn.cursor() as cur:
            _set_clob_inputs(cur, "name_embedding")
            cur.execute(generate_merge_vertex_sql(_config), {
                "vertex_id": vertex_id,
                "canonical_name": canonical_name,
                "vertex_type": vertex_type,
                "properties": properties_json,
                "source_doc": source_doc,
                "source_chunk": source_chunk,
                "confidence": confidence,
                "consensus_count": consensus_count,
                "extraction_run": _config.run_id,
                "name_embedding": _vector_literal(embedding),
            })
            action = "updated" if cur.rowcount == 2 else "created"
            cur.execute(generate_merge_vertex_doc_sql(_config), {
                "vertex_id": vertex_id,
                "doc_fingerprint": doc_fingerprint,
                "source_doc": source_doc,
                "source_chunk": source_chunk,
                "confidence": confidence,
                "consensus_count": consensus_count,
                "extraction_run": _config.run_id,
            })
            conn.commit()

    _tool_log(
        "graphwriter",
        f"merge_vertex completed; id={vertex_id}, action={action}"
    )
    return json.dumps({"vertex_id": vertex_id, "action": action})


@tool(description_mode="only_docstring")
def merge_edge(
    source_vertex_id: str,
    target_vertex_id: str,
    relationship_type: str,
    properties_json: str,
    doc_fingerprint: str,
    source_doc: str,
    source_chunk: int,
    confidence: float,
    consensus_count: int,
) -> str:
    """
    Idempotent upsert of an edge into the edge table.
    Generates embedding for the relationship description.

    Parameters:
        source_vertex_id (str): Source vertex ID.
        target_vertex_id (str): Target vertex ID.
        relationship_type (str): Relationship type label.
        properties_json (str): JSON properties string.
        doc_fingerprint (str): Source document fingerprint.
        source_doc (str): Source document filename.
        source_chunk (int): Source chunk index.
        confidence (float): Extraction confidence.
        consensus_count (int): Number of agreeing extractors.

    Returns:
        str: JSON confirmation.
    """
    from schema import generate_merge_edge_doc_sql, generate_merge_edge_sql

    _tool_log(
        "graphwriter",
        f"merge_edge started; {source_vertex_id}->{relationship_type}->{target_vertex_id}"
    )
    desc = f"{source_vertex_id} {relationship_type} {target_vertex_id}"
    embedding = _embed(desc)

    with _pool.acquire() as conn:
        with conn.cursor() as cur:
            _set_clob_inputs(cur, "description_embedding")
            cur.execute(generate_merge_edge_sql(_config), {
                "source_vertex_id": source_vertex_id,
                "target_vertex_id": target_vertex_id,
                "relationship_type": relationship_type,
                "properties": properties_json,
                "source_doc": source_doc,
                "source_chunk": source_chunk,
                "confidence": confidence,
                "consensus_count": consensus_count,
                "extraction_run": _config.run_id,
                "description_embedding": _vector_literal(embedding),
            })
            cur.execute(f"""
                SELECT edge_id
                FROM {_config.edge_table}
                WHERE source_vertex_id = :source_vertex_id
                  AND target_vertex_id = :target_vertex_id
                  AND relationship_type = :relationship_type
            """, {
                "source_vertex_id": source_vertex_id,
                "target_vertex_id": target_vertex_id,
                "relationship_type": relationship_type,
            })
            row = cur.fetchone()
            if not row:
                raise RuntimeError(
                    "Unable to locate edge after merge: "
                    f"{source_vertex_id}->{relationship_type}->{target_vertex_id}"
                )
            edge_id = row[0]
            cur.execute(generate_merge_edge_doc_sql(_config), {
                "edge_id": edge_id,
                "doc_fingerprint": doc_fingerprint,
                "source_doc": source_doc,
                "source_chunk": source_chunk,
                "confidence": confidence,
                "consensus_count": consensus_count,
                "extraction_run": _config.run_id,
            })
            conn.commit()

    _tool_log(
        "graphwriter",
        f"merge_edge completed; {source_vertex_id}->{relationship_type}->{target_vertex_id}"
    )
    return json.dumps({
        "edge_id": edge_id,
        "edge": f"{source_vertex_id}->{relationship_type}->{target_vertex_id}",
    })


@tool(description_mode="only_docstring")
def store_chunks(chunks_json: str) -> str:
    """
    Store document chunks with embeddings in the chunk table.

    Parameters:
        chunks_json (str): JSON list of chunks with text and metadata.

    Returns:
        str: JSON with count of stored chunks.
    """
    payload = json.loads(chunks_json)
    if isinstance(payload, list):
        chunks = payload
    elif isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
        chunks = payload["chunks"]
    elif isinstance(payload, dict) and payload.get("doc_fingerprint"):
        doc_fingerprint = payload["doc_fingerprint"]
        chunks = _document_chunk_cache.get(doc_fingerprint)
        if chunks is None:
            cached_payload = _load_chunk_cache(doc_fingerprint)
            chunks = cached_payload.get("chunks", []) if cached_payload else []
        if not chunks:
            raise RuntimeError(
                f"No cached chunks found for doc_fingerprint={doc_fingerprint}"
            )
    else:
        raise ValueError(
            "store_chunks expects a chunk list, a {'chunks': [...]} object, "
            "or a manifest with doc_fingerprint"
        )

    _tool_log("graphwriter", f"store_chunks started; chunks={len(chunks)}")
    texts = [c["text"] for c in chunks]
    embeddings = _embed_batch(texts)
    doc_fingerprints = sorted({
        str(chunk.get("metadata", {}).get("doc_fingerprint", "")).strip()
        for chunk in chunks
        if str(chunk.get("metadata", {}).get("doc_fingerprint", "")).strip()
    })

    with _pool.acquire() as conn:
        with conn.cursor() as cur:
            for doc_fingerprint in doc_fingerprints:
                cur.execute(
                    f"DELETE FROM {_config.chunk_table} "
                    "WHERE doc_fingerprint = :doc_fingerprint",
                    {"doc_fingerprint": doc_fingerprint},
                )

            _set_clob_inputs(cur, "chunk_text", "text_embedding")
            for i, chunk in enumerate(chunks):
                meta = chunk.get("metadata", {})
                cur.execute(f"""
                    INSERT INTO {_config.chunk_table}
                        (doc_fingerprint, source_file, chunk_index, chunk_text,
                         structural_type, page_or_section, text_embedding,
                         extraction_run)
                    VALUES (
                        :doc_fingerprint, :source_file, :chunk_index, :chunk_text,
                        :structural_type, :page_or_section,
                        TO_VECTOR(:text_embedding, {_config.openai.embedding_dimensions}, FLOAT32),
                        :extraction_run
                    )
                """, {
                    "doc_fingerprint": meta.get("doc_fingerprint", ""),
                    "source_file": meta.get("source_file", ""),
                    "chunk_index": meta.get("chunk_index", i),
                    "chunk_text": chunk["text"],
                    "structural_type": meta.get("structural_type", "paragraph"),
                    "page_or_section": meta.get("page_or_section", ""),
                    "text_embedding": _vector_literal(embeddings[i]),
                    "extraction_run": _config.run_id,
                })
            conn.commit()

    _tool_log("graphwriter", f"store_chunks completed; chunks={len(chunks)}")
    return json.dumps({"chunks_stored": len(chunks)})


@tool(description_mode="only_docstring")
def save_extraction_record(
    doc_fingerprint: str,
    filename: str,
    chunks_processed: int,
    vertices_created: int,
    edges_created: int,
    consensus_rate: float,
) -> str:
    """
    Save an extraction record to Agent Memory so the Parser
    can skip re-processing this document.

    In ontology-only Agent Memory mode this is a no-op; memory is reserved
    for ontology entries.

    Parameters:
        doc_fingerprint (str): SHA-256 fingerprint of the document.
        filename (str): Original filename.
        chunks_processed (int): Number of chunks processed.
        vertices_created (int): Count of vertices created.
        edges_created (int): Count of edges created.
        consensus_rate (float): Ratio of triples that reached consensus.

    Returns:
        str: Confirmation JSON.
    """
    _tool_log(
        "graphwriter",
        f"save_extraction_record started; filename={filename}, "
        f"chunks={chunks_processed}"
    )
    if _agent_memory_ontology_only():
        _tool_log("graphwriter", "save_extraction_record skipped; ontology-only memory")
        return json.dumps({"saved": False, "memory_usage": "ontology_only"})

    record = json.dumps({
        "memory_type": "extraction_record",
        "doc_fingerprint": doc_fingerprint,
        "filename": filename,
        "chunks_processed": chunks_processed,
        "vertices_created": vertices_created,
        "edges_created": edges_created,
        "consensus_rate": consensus_rate,
        "processed_at": datetime.now().isoformat(),
        "extraction_run": _config.run_id,
    })
    _memory.add_memory(
        record,
        user_id="system",
        agent_id="parser_agent",
    )
    _tool_log("graphwriter", "save_extraction_record completed")
    return json.dumps({"saved": True})


# ══════════════════════════════════════════════════════════════
# Auditor tools
# ══════════════════════════════════════════════════════════════

@tool(description_mode="only_docstring")
def validate_graph() -> str:
    """
    Run validation queries on the Property Graph:
    - Orphaned vertices (no edges)
    - Near-duplicate vertices (vector similarity > 0.90 within same type)
    - Low-consensus vertices (consensus_count = K minimum)

    Returns:
        str: JSON list of validation issues found.
    """
    issues = []

    with _pool.acquire() as conn:
        with conn.cursor() as cur:
            # Orphaned vertices
            cur.execute(f"""
                SELECT v.vertex_id, v.canonical_name, v.vertex_type
                FROM {_config.vertex_table} v
                WHERE NOT EXISTS (
                    SELECT 1 FROM {_config.edge_table} e
                    WHERE e.source_vertex_id = v.vertex_id
                       OR e.target_vertex_id = v.vertex_id
                )
                AND v.vertex_type != 'document'
            """)
            for row in cur.fetchall():
                issues.append({
                    "issue_type": "orphan",
                    "description": f"Vertex '{row[1]}' ({row[2]}) has no edges",
                    "affected_vertex_ids": [row[0]],
                    "severity": "warning",
                })

            # Low-consensus vertices
            cur.execute(f"""
                SELECT vertex_id, canonical_name, vertex_type, consensus_count
                FROM {_config.vertex_table}
                WHERE consensus_count = :1
                  AND extraction_run = :2
            """, [_config.ensemble.consensus_k, _config.run_id])
            for row in cur.fetchall():
                issues.append({
                    "issue_type": "low_consensus",
                    "description": (
                        f"Vertex '{row[1]}' ({row[2]}) has bare-minimum "
                        f"consensus ({row[3]}/{_config.ensemble.consensus_n})"
                    ),
                    "affected_vertex_ids": [row[0]],
                    "severity": "warning",
                })

    return json.dumps({"issues": issues, "count": len(issues)},
                      ensure_ascii=False)


@tool(description_mode="only_docstring")
def find_near_duplicates() -> str:
    """
    Find near-duplicate vertices within the same type using
    vector similarity > 0.90. These escaped reconciliation.

    Returns:
        str: JSON list of near-duplicate pairs.
    """
    duplicates = []

    with _pool.acquire() as conn:
        with conn.cursor() as cur:
            # Get distinct vertex types
            cur.execute(f"SELECT DISTINCT vertex_type FROM {_config.vertex_table}")
            vtypes = [row[0] for row in cur.fetchall()]

            for vtype in vtypes:
                cur.execute(f"""
                    SELECT a.vertex_id, a.canonical_name,
                           b.vertex_id, b.canonical_name,
                           VECTOR_DISTANCE(a.name_embedding,
                                          b.name_embedding, COSINE) AS dist
                    FROM {_config.vertex_table} a, {_config.vertex_table} b
                    WHERE a.vertex_type = :1
                      AND b.vertex_type = :1
                      AND a.vertex_id < b.vertex_id
                      AND VECTOR_DISTANCE(a.name_embedding,
                                         b.name_embedding, COSINE) < :2
                """, [vtype, 1.0 - _config.auditor.near_duplicate_threshold])
                for row in cur.fetchall():
                    similarity = round(1.0 - (row[4] or 0.0), 4)
                    duplicates.append({
                        "vertex_a": {"id": row[0], "name": row[1]},
                        "vertex_b": {"id": row[2], "name": row[3]},
                        "type": vtype,
                        "similarity": similarity,
                    })

    return json.dumps({"near_duplicates": duplicates,
                       "count": len(duplicates)}, ensure_ascii=False)


@tool(description_mode="only_docstring")
def save_schema_rule(rule_json: str) -> str:
    """
    Save a new schema validation rule.

    In ontology-only Agent Memory mode this is a no-op.

    Parameters:
        rule_json (str): JSON with rule_text, source_types, target_types, edge_type.

    Returns:
        str: Confirmation JSON.
    """
    if _agent_memory_ontology_only():
        return json.dumps({"saved": False, "memory_usage": "ontology_only"})

    _memory.add_memory(
        rule_json,
        user_id="system",
        agent_id="auditor_agent",
    )
    return json.dumps({"saved": True})


@tool(description_mode="only_docstring")
def read_schema_rules() -> str:
    """
    Read all schema validation rules.

    In ontology-only Agent Memory mode this returns an empty rule set.

    Returns:
        str: JSON list of schema rules.
    """
    from oracleagentmemory.apis.searchscope import SearchScope

    if _agent_memory_ontology_only():
        return json.dumps({
            "rules": [],
            "memory_usage": "ontology_only",
        }, ensure_ascii=False)

    results = _memory.search(
        query="schema_rule edge_type source_types target_types",
        scope=SearchScope(user_id="system", agent_id="auditor_agent"),
        max_results=50,
    )
    rules = []
    for r in (results or []):
        try:
            data = json.loads(r.content)
            if data.get("memory_type") == "schema_rule":
                rules.append(data)
        except (json.JSONDecodeError, AttributeError):
            continue
    return json.dumps({"rules": rules}, ensure_ascii=False)


@tool(description_mode="only_docstring")
def update_ontology(action: str, entry_json: str) -> str:
    """
    Admit or reject a pending entity/relationship type in the
    ontology registry.

    Parameters:
        action (str): "admit" or "reject".
        entry_json (str): JSON with the type definition.

    Returns:
        str: Confirmation JSON.
    """
    if action == "admit":
        _memory.add_memory(
            entry_json,
            user_id="system",
            agent_id="ontology_manager",
        )
        return json.dumps({"admitted": True})
    else:
        return json.dumps({"rejected": True, "reason": "Auditor rejected type"})


@tool(description_mode="only_docstring")
def read_consensus_log() -> str:
    """
    Read recent consensus log entries to analyze extractor
    disagreement patterns.

    In ontology-only Agent Memory mode this returns an empty log.

    Returns:
        str: JSON with recent consensus decisions.
    """
    from oracleagentmemory.apis.searchscope import SearchScope

    if _agent_memory_ontology_only():
        return json.dumps({
            "entries": [],
            "memory_usage": "ontology_only",
        }, ensure_ascii=False)

    results = _memory.search(
        query="consensus_result accepted rejected",
        scope=SearchScope(user_id="system", agent_id="consensus_merger"),
        max_results=50,
    )
    entries = []
    for r in (results or []):
        try:
            entries.append(json.loads(r.content))
        except (json.JSONDecodeError, AttributeError):
            continue
    return json.dumps({"entries": entries}, ensure_ascii=False)
