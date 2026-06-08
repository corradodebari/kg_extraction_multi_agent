"""
SQL schema for the KG Extraction Swarm.
Tables for vertices, edges, chunks, and the Property Graph DDL.
"""

import hashlib
import re


def generate_table_ddls(config) -> list[str]:
    """Generate table DDL with the configured database object prefix."""
    vertex_table = config.vertex_table
    edge_table = config.edge_table
    vertex_doc_table = config.vertex_doc_table
    edge_doc_table = config.edge_doc_table
    chunk_table = config.chunk_table
    vector_dimensions = config.openai.embedding_dimensions

    return [
        f"""
-- ============================================================
-- Vertex table
-- ============================================================
CREATE TABLE IF NOT EXISTS {vertex_table} (
    vertex_id       VARCHAR2(200) PRIMARY KEY,
    canonical_name  VARCHAR2(500) NOT NULL,
    vertex_type     VARCHAR2(100) NOT NULL,
    properties      JSON,
    source_doc      VARCHAR2(500),
    source_chunk    NUMBER,
    confidence      NUMBER(4,3),
    consensus_count NUMBER(2),
    extraction_run  VARCHAR2(100),
    merge_history   JSON DEFAULT JSON('[]'),
    name_embedding  VECTOR({vector_dimensions}, FLOAT32),
    created_at      TIMESTAMP DEFAULT SYSTIMESTAMP,
    updated_at      TIMESTAMP DEFAULT SYSTIMESTAMP,
    CONSTRAINT {config.db_object_name("UQ_VERTEX_NAME_TYPE")}
        UNIQUE (canonical_name, vertex_type)
)
""",
        f"""
CREATE TABLE IF NOT EXISTS {edge_table} (
    edge_id             NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_vertex_id    VARCHAR2(200) NOT NULL,
    target_vertex_id    VARCHAR2(200) NOT NULL,
    relationship_type   VARCHAR2(100) NOT NULL,
    properties          JSON,
    source_doc          VARCHAR2(500),
    source_chunk        NUMBER,
    confidence          NUMBER(4,3),
    consensus_count     NUMBER(2),
    extraction_run      VARCHAR2(100),
    description_embedding VECTOR({vector_dimensions}, FLOAT32),
    created_at          TIMESTAMP DEFAULT SYSTIMESTAMP,
    CONSTRAINT {config.db_object_name("FK_EDGE_SOURCE")}
        FOREIGN KEY (source_vertex_id) REFERENCES {vertex_table}(vertex_id),
    CONSTRAINT {config.db_object_name("FK_EDGE_TARGET")}
        FOREIGN KEY (target_vertex_id) REFERENCES {vertex_table}(vertex_id),
    CONSTRAINT {config.db_object_name("UQ_EDGE")}
        UNIQUE (source_vertex_id, relationship_type, target_vertex_id)
)
""",
        f"""
CREATE TABLE IF NOT EXISTS {vertex_doc_table} (
    vertex_id       VARCHAR2(200) NOT NULL,
    doc_fingerprint VARCHAR2(100) NOT NULL,
    source_doc      VARCHAR2(500),
    source_chunk    NUMBER NOT NULL,
    confidence      NUMBER(4,3),
    consensus_count NUMBER(2),
    extraction_run  VARCHAR2(100),
    created_at      TIMESTAMP DEFAULT SYSTIMESTAMP,
    updated_at      TIMESTAMP DEFAULT SYSTIMESTAMP,
    CONSTRAINT {config.db_object_name("FK_VERTEX_DOC_VERTEX")}
        FOREIGN KEY (vertex_id) REFERENCES {vertex_table}(vertex_id),
    CONSTRAINT {config.db_object_name("UQ_VERTEX_DOC")}
        UNIQUE (vertex_id, doc_fingerprint, source_chunk)
)
""",
        f"""
CREATE TABLE IF NOT EXISTS {edge_doc_table} (
    edge_id         NUMBER NOT NULL,
    doc_fingerprint VARCHAR2(100) NOT NULL,
    source_doc      VARCHAR2(500),
    source_chunk    NUMBER NOT NULL,
    confidence      NUMBER(4,3),
    consensus_count NUMBER(2),
    extraction_run  VARCHAR2(100),
    created_at      TIMESTAMP DEFAULT SYSTIMESTAMP,
    updated_at      TIMESTAMP DEFAULT SYSTIMESTAMP,
    CONSTRAINT {config.db_object_name("FK_EDGE_DOC_EDGE")}
        FOREIGN KEY (edge_id) REFERENCES {edge_table}(edge_id),
    CONSTRAINT {config.db_object_name("UQ_EDGE_DOC")}
        UNIQUE (edge_id, doc_fingerprint, source_chunk)
)
""",
        f"""
CREATE TABLE IF NOT EXISTS {chunk_table} (
    chunk_id        NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_fingerprint VARCHAR2(100) NOT NULL,
    source_file     VARCHAR2(500),
    chunk_index     NUMBER NOT NULL,
    chunk_text      CLOB,
    structural_type VARCHAR2(50),
    page_or_section VARCHAR2(200),
    text_embedding  VECTOR({vector_dimensions}, FLOAT32),
    extraction_run  VARCHAR2(100),
    created_at      TIMESTAMP DEFAULT SYSTIMESTAMP
)
""",
    ]


def generate_index_ddls(config) -> list[str]:
    """Generate index DDL with the configured database object prefix."""
    vertex_table = config.vertex_table
    edge_table = config.edge_table
    vertex_doc_table = config.vertex_doc_table
    edge_doc_table = config.edge_doc_table
    chunk_table = config.chunk_table

    return [
        f"CREATE INDEX IF NOT EXISTS {config.db_object_name('IDX_VERTEX_TYPE')} "
        f"ON {vertex_table}(vertex_type)",
        f"CREATE INDEX IF NOT EXISTS {config.db_object_name('IDX_VERTEX_NAME')} "
        f"ON {vertex_table}(canonical_name)",
        f"CREATE INDEX IF NOT EXISTS {config.db_object_name('IDX_EDGE_SOURCE')} "
        f"ON {edge_table}(source_vertex_id)",
        f"CREATE INDEX IF NOT EXISTS {config.db_object_name('IDX_EDGE_TARGET')} "
        f"ON {edge_table}(target_vertex_id)",
        f"CREATE INDEX IF NOT EXISTS {config.db_object_name('IDX_EDGE_TYPE')} "
        f"ON {edge_table}(relationship_type)",
        f"CREATE INDEX IF NOT EXISTS {config.db_object_name('IDX_VERTEX_DOC_DOC')} "
        f"ON {vertex_doc_table}(doc_fingerprint)",
        f"CREATE INDEX IF NOT EXISTS {config.db_object_name('IDX_EDGE_DOC_DOC')} "
        f"ON {edge_doc_table}(doc_fingerprint)",
        f"CREATE INDEX IF NOT EXISTS {config.db_object_name('IDX_CHUNK_FINGERPRINT')} "
        f"ON {chunk_table}(doc_fingerprint)",
        f"""
CREATE VECTOR INDEX IF NOT EXISTS {config.db_object_name("IDX_VERTEX_VEC")}
    ON {vertex_table}(name_embedding)
    ORGANIZATION INMEMORY NEIGHBOR GRAPH
    DISTANCE COSINE
    WITH TARGET ACCURACY 95
""",
        f"""
CREATE VECTOR INDEX IF NOT EXISTS {config.db_object_name("IDX_CHUNK_VEC")}
    ON {chunk_table}(text_embedding)
    ORGANIZATION INMEMORY NEIGHBOR GRAPH
    DISTANCE COSINE
    WITH TARGET ACCURACY 95
""",
    ]


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _graph_label(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _ontology_labels(items: list[dict], fallback: list[dict]) -> list[str]:
    labels: list[str] = []
    for item in items or fallback:
        label = str(item.get("label", "")).strip().lower()
        if label and label not in labels:
            labels.append(label)
    return labels


def _ontology_items(items: list[dict], fallback: list[dict]) -> list[dict]:
    seen: set[str] = set()
    normalized_items: list[dict] = []
    by_label: dict[str, dict] = {}
    for item in items or fallback:
        label = str(item.get("label", "")).strip().lower()
        if not label:
            continue
        normalized = dict(item)
        normalized["label"] = label
        normalized["source"] = str(normalized.get("source", "any") or "any").strip().lower()
        normalized["target"] = str(normalized.get("target", "any") or "any").strip().lower()
        if label not in seen:
            normalized_items.append(normalized)
            by_label[label] = normalized
            seen.add(label)
            continue

        current = by_label[label]
        if current.get("source", "any") != normalized.get("source", "any"):
            current["source"] = "any"
        if current.get("target", "any") != normalized.get("target", "any"):
            current["target"] = "any"
    return normalized_items


def _object_suffix(*parts: str, max_length: int = 96) -> str:
    raw = "_".join(str(part).strip().upper() for part in parts if str(part).strip())
    suffix = re.sub(r"[^A-Z0-9_]+", "_", raw).strip("_")
    if not suffix:
        suffix = "X"
    if len(suffix) <= max_length:
        return suffix
    digest = hashlib.sha1(suffix.encode("utf-8")).hexdigest()[:10].upper()
    return f"{suffix[:max_length - 11]}_{digest}"


def typed_vertex_mv_name(config, vertex_type: str) -> str:
    return config.db_object_name(f"KG_V_{_object_suffix(vertex_type)}")


def typed_edge_mv_name(config, relationship_type: str) -> str:
    suffix = _object_suffix(relationship_type)
    return config.db_object_name(f"KG_E_{suffix}")


def typed_graph_materialized_view_names(
    config,
    vertex_types: list[dict],
    edge_types: list[dict],
) -> list[str]:
    vertex_labels = _ontology_labels(vertex_types, DEFAULT_VERTEX_TYPES)
    edge_items = _ontology_items(edge_types, DEFAULT_EDGE_TYPES)
    names = [typed_vertex_mv_name(config, label) for label in vertex_labels]
    for edge_item in edge_items:
        names.append(typed_edge_mv_name(config, edge_item["label"]))
    return names


def generate_typed_graph_materialized_view_ddls(
    config,
    vertex_types: list[dict],
    edge_types: list[dict],
) -> list[str]:
    """
    Generate typed materialized views used by the property graph.

    Oracle SQL property graphs can color/style by labels, but row-level label
    filters are not valid in this database version. Materialized views give
    each vertex_type and relationship_type a dedicated graph element source.
    This intentionally creates one graph-visible table per vertex type and
    one graph-visible table per edge type, matching the GraphRAG builder shape.
    """
    vertex_labels = _ontology_labels(vertex_types, DEFAULT_VERTEX_TYPES)
    edge_items = _ontology_items(edge_types, DEFAULT_EDGE_TYPES)
    ddls: list[str] = []

    for vertex_label in vertex_labels:
        ddls.append(f"""
CREATE MATERIALIZED VIEW {typed_vertex_mv_name(config, vertex_label)} AS
SELECT vertex_id, canonical_name, vertex_type, properties, source_doc,
       source_chunk, confidence, consensus_count, extraction_run
FROM {config.vertex_table}
WHERE vertex_type = {_sql_string_literal(vertex_label)}
""")

    for edge_item in edge_items:
        edge_label = edge_item["label"]
        ddls.append(f"""
CREATE MATERIALIZED VIEW {typed_edge_mv_name(config, edge_label)} AS
SELECT e.edge_id, e.source_vertex_id, e.target_vertex_id,
       e.relationship_type, e.properties, e.source_doc, e.source_chunk,
       e.confidence, e.consensus_count, e.extraction_run
FROM {config.edge_table} e
WHERE e.relationship_type = {_sql_string_literal(edge_label)}
""")

    return ddls


def _edge_endpoint_type(edge_type: dict, endpoint: str, vertex_labels: list[str]) -> str | None:
    label = str(edge_type.get(endpoint, "any") or "any").strip().lower()
    if label in {"", "any", "*"}:
        return None
    if label in vertex_labels:
        return label
    return None


def generate_property_graph_ddl(
    config,
    vertex_types: list[dict],
    edge_types: list[dict],
) -> str:
    """
    Generate CREATE PROPERTY GRAPH DDL from the ontology registry.
    Called by the GraphWriter on first run.

    The graph is built over typed materialized views so SQL Developer Graph
    Visualization can style vertex and edge labels by type. It follows the
    GraphRAG builder layout: one graph-visible table per vertex type and one
    graph-visible table per edge type.
    """
    vertex_labels = _ontology_labels(vertex_types, DEFAULT_VERTEX_TYPES)
    edge_items = _ontology_items(edge_types, DEFAULT_EDGE_TYPES)
    needs_entity_vertex = any(
        _edge_endpoint_type(edge_item, "source", vertex_labels) is None
        or _edge_endpoint_type(edge_item, "target", vertex_labels) is None
        for edge_item in edge_items
    )

    vertex_clauses = []
    if needs_entity_vertex:
        vertex_clauses.append(f"""
        {config.vertex_table} AS V_ENTITY
            KEY (vertex_id)
            LABEL {_graph_label("entity")}
            PROPERTIES (
                vertex_id, canonical_name, vertex_type, properties, source_doc,
                source_chunk, confidence, consensus_count, extraction_run
            )""")

    for vertex_label in vertex_labels:
        mv_name = typed_vertex_mv_name(config, vertex_label)
        element_name = f"V_{_object_suffix(vertex_label, max_length=80)}"
        vertex_clauses.append(f"""
        {mv_name} AS {element_name}
            KEY (vertex_id)
            LABEL {_graph_label(vertex_label)}
            PROPERTIES (
                vertex_id, canonical_name, vertex_type, properties, source_doc,
                source_chunk, confidence, consensus_count, extraction_run
            )""")

    edge_clauses = []
    for edge_item in edge_items:
        edge_label = edge_item["label"]
        source_label = _edge_endpoint_type(edge_item, "source", vertex_labels)
        target_label = _edge_endpoint_type(edge_item, "target", vertex_labels)
        source_element = (
            f"V_{_object_suffix(source_label, max_length=80)}"
            if source_label else "V_ENTITY"
        )
        target_element = (
            f"V_{_object_suffix(target_label, max_length=80)}"
            if target_label else "V_ENTITY"
        )
        mv_name = typed_edge_mv_name(config, edge_label)
        element_name = f"E_{_object_suffix(edge_label, max_length=80)}"
        edge_clauses.append(f"""
        {mv_name} AS {element_name}
            KEY (edge_id)
            SOURCE KEY (source_vertex_id) REFERENCES {source_element}(vertex_id)
            DESTINATION KEY (target_vertex_id) REFERENCES {target_element}(vertex_id)
            LABEL {_graph_label(edge_label)}
            PROPERTIES (
                edge_id, source_vertex_id, target_vertex_id, relationship_type,
                properties, source_doc, source_chunk, confidence,
                consensus_count, extraction_run
            )""")

    return f"""
CREATE PROPERTY GRAPH {config.graph_name}
    VERTEX TABLES (
{",".join(vertex_clauses)}
    )
    EDGE TABLES (
{",".join(edge_clauses)}
    )
"""


def generate_merge_vertex_sql(config) -> str:
    """Generate idempotent vertex upsert SQL."""
    vector_dimensions = config.openai.embedding_dimensions
    return f"""
MERGE INTO {config.vertex_table} tgt
USING (SELECT :vertex_id AS vid, :canonical_name AS vname,
              :vertex_type AS vtype FROM DUAL) src
ON (tgt.canonical_name = src.vname AND tgt.vertex_type = src.vtype)
WHEN MATCHED THEN UPDATE SET
    tgt.properties = JSON_MERGEPATCH(tgt.properties, :properties),
    tgt.confidence = GREATEST(tgt.confidence, :confidence),
    tgt.consensus_count = GREATEST(tgt.consensus_count, :consensus_count),
    tgt.updated_at = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT
    (vertex_id, canonical_name, vertex_type, properties, source_doc,
     source_chunk, confidence, consensus_count, extraction_run,
     name_embedding, merge_history)
VALUES
    (:vertex_id, :canonical_name, :vertex_type, :properties, :source_doc,
     :source_chunk, :confidence, :consensus_count, :extraction_run,
     TO_VECTOR(:name_embedding, {vector_dimensions}, FLOAT32), JSON('[]'))
"""


def generate_merge_edge_sql(config) -> str:
    """Generate idempotent edge upsert SQL."""
    vector_dimensions = config.openai.embedding_dimensions
    return f"""
MERGE INTO {config.edge_table} tgt
USING (SELECT :source_vertex_id AS svid, :target_vertex_id AS tvid,
              :relationship_type AS rtype FROM DUAL) src
ON (tgt.source_vertex_id = src.svid
    AND tgt.target_vertex_id = src.tvid
    AND tgt.relationship_type = src.rtype)
WHEN MATCHED THEN UPDATE SET
    tgt.confidence = GREATEST(tgt.confidence, :confidence),
    tgt.consensus_count = GREATEST(tgt.consensus_count, :consensus_count)
WHEN NOT MATCHED THEN INSERT
    (source_vertex_id, target_vertex_id, relationship_type, properties,
     source_doc, source_chunk, confidence, consensus_count,
     extraction_run, description_embedding)
VALUES
    (:source_vertex_id, :target_vertex_id, :relationship_type, :properties,
     :source_doc, :source_chunk, :confidence, :consensus_count,
     :extraction_run, TO_VECTOR(:description_embedding, {vector_dimensions}, FLOAT32))
"""


def generate_merge_vertex_doc_sql(config) -> str:
    """Generate idempotent vertex-document provenance upsert SQL."""
    return f"""
MERGE INTO {config.vertex_doc_table} tgt
USING (SELECT :vertex_id AS vid, :doc_fingerprint AS docfp,
              :source_chunk AS chunk_idx FROM DUAL) src
ON (tgt.vertex_id = src.vid
    AND tgt.doc_fingerprint = src.docfp
    AND tgt.source_chunk = src.chunk_idx)
WHEN MATCHED THEN UPDATE SET
    tgt.confidence = GREATEST(tgt.confidence, :confidence),
    tgt.consensus_count = GREATEST(tgt.consensus_count, :consensus_count),
    tgt.extraction_run = :extraction_run,
    tgt.updated_at = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT
    (vertex_id, doc_fingerprint, source_doc, source_chunk, confidence,
     consensus_count, extraction_run)
VALUES
    (:vertex_id, :doc_fingerprint, :source_doc, :source_chunk, :confidence,
     :consensus_count, :extraction_run)
"""


def generate_merge_edge_doc_sql(config) -> str:
    """Generate idempotent edge-document provenance upsert SQL."""
    return f"""
MERGE INTO {config.edge_doc_table} tgt
USING (SELECT :edge_id AS eid, :doc_fingerprint AS docfp,
              :source_chunk AS chunk_idx FROM DUAL) src
ON (tgt.edge_id = src.eid
    AND tgt.doc_fingerprint = src.docfp
    AND tgt.source_chunk = src.chunk_idx)
WHEN MATCHED THEN UPDATE SET
    tgt.confidence = GREATEST(tgt.confidence, :confidence),
    tgt.consensus_count = GREATEST(tgt.consensus_count, :consensus_count),
    tgt.extraction_run = :extraction_run,
    tgt.updated_at = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT
    (edge_id, doc_fingerprint, source_doc, source_chunk, confidence,
     consensus_count, extraction_run)
VALUES
    (:edge_id, :doc_fingerprint, :source_doc, :source_chunk, :confidence,
     :consensus_count, :extraction_run)
"""


# Default ontology seed (used on first run)
DEFAULT_VERTEX_TYPES = [
    {"label": "person", "description": "Individual human",
     "naming_convention": "Full name, title case"},
    {"label": "organization", "description": "Company, institution, agency",
     "naming_convention": "Official registered name, title case"},
    {"label": "technology", "description": "Software, hardware, protocol, standard",
     "naming_convention": "Official product/technology name"},
    {"label": "concept", "description": "Abstract idea, methodology, theory",
     "naming_convention": "Lowercase unless proper noun"},
    {"label": "document", "description": "Source document (provenance)",
     "naming_convention": "Document title"},
    {"label": "location", "description": "Geographic entity",
     "naming_convention": "Official geographic name"},
    {"label": "event", "description": "Conference, release, incident",
     "naming_convention": "Event name, title case"},
]

DEFAULT_EDGE_TYPES = [
    {"label": "works_for", "source": "person", "target": "organization"},
    {"label": "founded", "source": "person", "target": "organization"},
    {"label": "subsidiary_of", "source": "organization", "target": "organization"},
    {"label": "uses", "source": "organization", "target": "technology"},
    {"label": "developed_by", "source": "technology", "target": "organization"},
    {"label": "related_to", "source": "any", "target": "any"},
    {"label": "located_in", "source": "organization", "target": "location"},
    {"label": "participated_in", "source": "any", "target": "event"},
    {"label": "mentioned_in", "source": "any", "target": "document"},
    {"label": "implements", "source": "technology", "target": "concept"},
]
