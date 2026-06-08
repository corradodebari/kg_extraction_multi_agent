"""
Shared data models for the KG Extraction Swarm.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class ChunkMetadata:
    source_file: str
    page_or_section: str
    doc_fingerprint: str
    chunk_index: int
    structural_type: str  # "paragraph", "table", "heading"


@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata


@dataclass
class Entity:
    name: str
    entity_type: str
    properties: dict = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class Triple:
    subject: Entity
    relationship: str
    object: Entity
    confidence: float = 1.0
    source_chunk_index: int = -1
    source_doc: str = ""
    pending_types: List[str] = field(default_factory=list)


@dataclass
class ExtractorResult:
    """Output from a single extractor for one document."""
    extractor_name: str  # "alpha", "beta", "gamma"
    triples: List[Triple] = field(default_factory=list)


@dataclass
class ConsensusTriple:
    """A triple that passed consensus filtering."""
    subject: Entity
    relationship: str
    object: Entity
    consensus_count: int         # how many extractors agreed
    agreeing_extractors: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source_chunk_index: int = -1
    source_doc: str = ""
    pending_types: List[str] = field(default_factory=list)


@dataclass
class ReconciledTriple:
    """A triple after dedup and canonicalization."""
    subject_canonical: str
    subject_type: str
    subject_vertex_id: str
    relationship: str
    object_canonical: str
    object_type: str
    object_vertex_id: str
    consensus_count: int
    confidence: float
    source_doc: str
    source_chunk_index: int
    extraction_run: str


@dataclass
class WriteReport:
    """Report from the GraphWriter."""
    vertices_created: int = 0
    vertices_updated: int = 0
    edges_created: int = 0
    edges_updated: int = 0
    chunks_stored: int = 0
    doc_fingerprint: str = ""
    extraction_run: str = ""


@dataclass
class ValidationIssue:
    issue_type: str  # "orphan", "type_violation", "near_duplicate", "low_consensus"
    description: str
    affected_vertex_ids: List[str] = field(default_factory=list)
    severity: str = "warning"  # "warning", "error"
