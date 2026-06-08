"""
Configuration for the KG extraction pipeline.
All LLM calls go through an OpenAI-compatible API endpoint.
"""

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file without overriding env vars."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        try:
            parts = shlex.split(raw_line, comments=True, posix=True)
        except ValueError:
            continue

        if not parts:
            continue

        if parts[0] == "export":
            parts = parts[1:]

        if len(parts) != 1 or "=" not in parts[0]:
            continue

        key, value = parts[0].split("=", 1)
        key = key.strip()
        if key and all(char.isalnum() or char == "_" for char in key):
            os.environ.setdefault(key, value)


def _load_env() -> None:
    custom_env = os.getenv("GRAPH_SWARM_ENV_FILE")
    if custom_env:
        _load_env_file(Path(custom_env).expanduser())
        return

    _load_env_file(Path.cwd() / ".env")
    _load_env_file(Path(__file__).resolve().parent / ".env")


_load_env()


def _validate_db_identifier(value: str, setting_name: str) -> str:
    value = value.strip().upper()
    if not value:
        raise ValueError(f"{setting_name} cannot be empty")

    valid_chars = all(char.isalnum() or char == "_" for char in value)
    if not value[0].isalpha() or not valid_chars:
        raise ValueError(
            f"{setting_name} must start with a letter and contain only "
            "letters, numbers, and underscores"
        )

    if len(value) > 128:
        raise ValueError(f"{setting_name} must be 128 characters or fewer")

    return value


def _validate_db_prefix(value: str) -> str:
    value = value.strip().upper()
    if not value:
        return ""

    return _validate_db_identifier(value, "DB_OBJECT_PREFIX")


def _db_object_prefix() -> str:
    return _validate_db_prefix(os.getenv("DB_OBJECT_PREFIX", ""))


def _db_object_name(base_name: str, prefix: str | None = None) -> str:
    prefix = _db_object_prefix() if prefix is None else _validate_db_prefix(prefix)
    name = f"{prefix}{_validate_db_identifier(base_name, 'database object name')}"
    if len(name) > 128:
        raise ValueError(
            f"database object name '{name}' is too long; shorten DB_OBJECT_PREFIX"
        )
    return name


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no"}


def _env_csv(name: str) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _cycle_values(values: tuple[str, ...], count: int) -> list[str]:
    return [values[index % len(values)] for index in range(count)]


def _require_csv_count(name: str, values: list[str], count: int) -> None:
    if len(values) != count:
        raise ValueError(
            f"{name} must contain exactly {count} comma-separated values"
        )


@dataclass
class DBConfig:
    user: str = os.getenv("ORACLE_USER", "kg_swarm_user")
    password: str = os.getenv("ORACLE_PASSWORD", "YourPassword123")
    dsn: str = os.getenv("ORACLE_DSN", "localhost:1521/FREEPDB1")
    pool_min: int = 2
    pool_max: int = 10


@dataclass
class OpenAIConfig:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("OPENAI_BASE_URL", "").strip()
    timeout_seconds: float = _env_float("OPENAI_TIMEOUT_SECONDS", 300.0)
    max_retries: int = _env_int("OPENAI_MAX_RETRIES", 2)
    embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "openai/text-embedding-3-large",
    )
    embedding_dimensions: int = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "3072"))

    @property
    def embedding_provider_model(self) -> str:
        if "/" in self.embedding_model:
            return self.embedding_model
        return f"openai/{self.embedding_model}"

    @property
    def embedding_api_model(self) -> str:
        if self.embedding_model.startswith("openai/"):
            return self.embedding_model.removeprefix("openai/")
        return self.embedding_model

@dataclass
class ExtractorVariant:
    """Configuration for one extractor in the ensemble."""
    name: str
    model: str
    temperature: float
    prompt_angle: str  # "entity_first", "relationship_first", "balanced"
    agent_id: str = ""

    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = (
                self.name if self.name.startswith("extractor_")
                else f"extractor_{self.name}"
            )


DEFAULT_EXTRACTOR_MODELS = ("gpt-4o", "gpt-4o-mini", "gpt-4o")
DEFAULT_EXTRACTOR_TEMPERATURES = ("0.0", "0.0", "0.0")
DEFAULT_EXTRACTOR_PROMPT_ANGLES = (
    "entity_first",
    "relationship_first",
    "balanced",
)
VALID_EXTRACTOR_PROMPT_ANGLES = set(DEFAULT_EXTRACTOR_PROMPT_ANGLES)


def _build_extractor_variants() -> list[ExtractorVariant]:
    models = _env_csv("EXTRACTOR_MODELS")
    temperatures_raw = _env_csv("EXTRACTOR_TEMPERATURES")
    prompt_angles = _env_csv("EXTRACTOR_PROMPT_ANGLES")

    if os.getenv("EXTRACTOR_COUNT") is not None:
        count = _env_int("EXTRACTOR_COUNT", 3)
    elif models:
        count = len(models)
    elif temperatures_raw:
        count = len(temperatures_raw)
    else:
        count = 3

    if count < 1:
        raise ValueError("EXTRACTOR_COUNT must be at least 1")

    if models:
        _require_csv_count("EXTRACTOR_MODELS", models, count)
    else:
        models = _cycle_values(DEFAULT_EXTRACTOR_MODELS, count)

    if temperatures_raw:
        _require_csv_count("EXTRACTOR_TEMPERATURES", temperatures_raw, count)
    else:
        temperatures_raw = _cycle_values(DEFAULT_EXTRACTOR_TEMPERATURES, count)

    if prompt_angles:
        if len(prompt_angles) != count:
            prompt_angles = [
                prompt_angles[index % len(prompt_angles)]
                for index in range(count)
            ]
    else:
        prompt_angles = _cycle_values(DEFAULT_EXTRACTOR_PROMPT_ANGLES, count)

    temperatures = []
    for raw in temperatures_raw:
        try:
            temperatures.append(float(raw))
        except ValueError as exc:
            raise ValueError("EXTRACTOR_TEMPERATURES must contain numbers") from exc

    for prompt_angle in prompt_angles:
        if prompt_angle not in VALID_EXTRACTOR_PROMPT_ANGLES:
            allowed = ", ".join(sorted(VALID_EXTRACTOR_PROMPT_ANGLES))
            raise ValueError(
                "EXTRACTOR_PROMPT_ANGLES values must be one of: "
                f"{allowed}"
            )

    return [
        ExtractorVariant(
            name=f"extractor_{index + 1}",
            model=models[index],
            temperature=temperatures[index],
            prompt_angle=prompt_angles[index],
        )
        for index in range(count)
    ]


@dataclass
class EnsembleConfig:
    """Ensemble extraction configuration."""
    variants: List[ExtractorVariant] = field(default_factory=_build_extractor_variants)
    consensus_min_agreement: int = _env_int("CONSENSUS_MIN_AGREEMENT", 2)
    consensus_n: int = 0
    name_similarity_threshold: float = float(
        os.getenv("CONSENSUS_NAME_SIMILARITY_THRESHOLD", "0.88")
    )
    extraction_batch_size: int = int(os.getenv("EXTRACTION_BATCH_SIZE", "30"))
    parallel_workers: int = int(os.getenv("EXTRACTOR_PARALLEL_WORKERS", "0"))
    strict_ontology: bool = _env_bool("EXTRACTION_STRICT_ONTOLOGY", True)

    @property
    def n(self) -> int:
        return len(self.variants)

    @property
    def consensus_k(self) -> int:
        return self.consensus_min_agreement


@dataclass
class ReconcilerConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    similarity_threshold: float = 0.85


@dataclass
class AuditorConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    near_duplicate_threshold: float = 0.90


@dataclass
class ChunkingConfig:
    max_tokens: int = 1500
    overlap_tokens: int = 200
    cache_enabled: bool = os.getenv("CHUNK_CACHE_ENABLED", "true").lower() not in {
        "0",
        "false",
        "no",
    }
    cache_dir: str = os.getenv(
        "CHUNK_CACHE_DIR",
        str(Path(__file__).resolve().parent / "cache" / "chunks"),
    )


@dataclass
class OntologyConfig:
    auto_admit_threshold: int = int(os.getenv("ONTOLOGY_AUTO_ADMIT_THRESHOLD", "3"))
    candidate_min_extractors: int = _env_int("ONTOLOGY_CANDIDATE_MIN_EXTRACTORS", 0)
    domains: str = os.getenv("ONTOLOGY_DOMAINS", "core")
    domain_config_file: str = os.getenv(
        "DOMAIN_CONFIG_FILE",
        str(Path(__file__).resolve().parent / "domain_config.json"),
    )


@dataclass
class AgentMemoryConfig:
    table_prefix: str = field(
        default_factory=lambda: os.getenv(
            "AGENT_MEMORY_TABLE_PREFIX",
            _db_object_name("KG_SWARM_"),
        )
    )
    schema_policy: str = os.getenv(
        "AGENT_MEMORY_SCHEMA_POLICY",
        "CREATE_IF_NECESSARY",
    )
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    ontology_only: bool = os.getenv(
        "AGENT_MEMORY_ONTOLOGY_ONLY",
        "true",
    ).lower() not in {"0", "false", "no"}


@dataclass
class Config:
    db_object_prefix: str = field(default_factory=_db_object_prefix)
    db: DBConfig = field(default_factory=DBConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    reconciler: ReconcilerConfig = field(default_factory=ReconcilerConfig)
    auditor: AuditorConfig = field(default_factory=AuditorConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    memory: AgentMemoryConfig = field(default_factory=AgentMemoryConfig)
    graph_name: str = field(
        default_factory=lambda: os.getenv("GRAPH_NAME", "KG_EXTRACTION_GRAPH")
    )
    run_id: str = ""  # set at runtime

    def __post_init__(self):
        self.db_object_prefix = _validate_db_prefix(self.db_object_prefix)
        self.graph_name = _db_object_name(self.graph_name, self.db_object_prefix)
        self.memory.table_prefix = _validate_db_identifier(
            self.memory.table_prefix,
            "AGENT_MEMORY_TABLE_PREFIX",
        )
        self.memory.schema_policy = self.memory.schema_policy.strip().upper()
        valid_memory_schema_policies = {
            "CREATE_IF_EMPTY",
            "CREATE_IF_NECESSARY",
            "RECREATE",
            "REQUIRE_EXISTING",
        }
        if self.memory.schema_policy not in valid_memory_schema_policies:
            allowed = ", ".join(sorted(valid_memory_schema_policies))
            raise ValueError(
                "AGENT_MEMORY_SCHEMA_POLICY must be one of: "
                f"{allowed}"
            )
        if self.openai.timeout_seconds <= 0:
            raise ValueError("OPENAI_TIMEOUT_SECONDS must be greater than 0")
        if self.openai.max_retries < 0:
            raise ValueError("OPENAI_MAX_RETRIES cannot be negative")
        if not self.ensemble.variants:
            raise ValueError("At least one extractor variant is required")
        self.ensemble.consensus_n = len(self.ensemble.variants)
        if self.ensemble.parallel_workers < 1:
            self.ensemble.parallel_workers = len(self.ensemble.variants)
        if self.ontology.candidate_min_extractors < 1:
            self.ontology.candidate_min_extractors = min(
                self.ensemble.consensus_min_agreement,
                self.ensemble.consensus_n,
            )
        if self.ensemble.extraction_batch_size < 1:
            raise ValueError("EXTRACTION_BATCH_SIZE must be at least 1")
        if self.ensemble.consensus_n < 1:
            raise ValueError("EXTRACTOR_COUNT must be at least 1")
        if self.ensemble.consensus_min_agreement < 1:
            raise ValueError("CONSENSUS_MIN_AGREEMENT must be at least 1")
        if self.ensemble.consensus_min_agreement > self.ensemble.consensus_n:
            raise ValueError(
                "CONSENSUS_MIN_AGREEMENT cannot be greater than "
                "EXTRACTOR_COUNT"
            )
        if not 0.0 <= self.ensemble.name_similarity_threshold <= 1.0:
            raise ValueError(
                "CONSENSUS_NAME_SIMILARITY_THRESHOLD must be between 0.0 and 1.0"
            )
        if self.ontology.auto_admit_threshold < 1:
            raise ValueError("ONTOLOGY_AUTO_ADMIT_THRESHOLD must be at least 1")
        if self.ontology.candidate_min_extractors < 1:
            raise ValueError("ONTOLOGY_CANDIDATE_MIN_EXTRACTORS must be at least 1")
        if self.ontology.candidate_min_extractors > self.ensemble.consensus_n:
            raise ValueError(
                "ONTOLOGY_CANDIDATE_MIN_EXTRACTORS cannot be greater than "
                "EXTRACTOR_COUNT"
            )

    def db_object_name(self, base_name: str) -> str:
        return _db_object_name(base_name, self.db_object_prefix)

    @property
    def vertex_table(self) -> str:
        return self.db_object_name("KG_VERTEX")

    @property
    def edge_table(self) -> str:
        return self.db_object_name("KG_EDGE")

    @property
    def vertex_doc_table(self) -> str:
        return self.db_object_name("KG_VERTEX_DOC")

    @property
    def edge_doc_table(self) -> str:
        return self.db_object_name("KG_EDGE_DOC")

    @property
    def chunk_table(self) -> str:
        return self.db_object_name("KG_CHUNK")
