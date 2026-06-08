"""
Domain configuration loading for document classification and ontology entries.
"""

import json
from pathlib import Path
from typing import Any


DEFAULT_DOMAIN_CONFIG_PATH = Path(__file__).resolve().parent / "domain_config.json"


def _domain_key(value: str) -> str:
    return str(value or "").strip().lower()


def load_domain_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or DEFAULT_DOMAIN_CONFIG_PATH).expanduser()
    if not config_path.is_absolute() and not config_path.exists():
        config_path = DEFAULT_DOMAIN_CONFIG_PATH.parent / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    domains = payload.get("domains")
    if not isinstance(domains, dict) or not domains:
        raise ValueError(f"Domain config {config_path} must define a non-empty domains object")

    normalized_domains: dict[str, dict[str, Any]] = {}
    for domain_name, domain_spec in domains.items():
        key = _domain_key(domain_name)
        if not key or not isinstance(domain_spec, dict):
            continue
        normalized = dict(domain_spec)
        normalized["keywords"] = [
            str(keyword).strip().lower()
            for keyword in normalized.get("keywords", [])
            if str(keyword).strip()
        ]
        normalized["vertex_types"] = [
            dict(item)
            for item in normalized.get("vertex_types", [])
            if isinstance(item, dict)
        ]
        normalized["edge_types"] = [
            dict(item)
            for item in normalized.get("edge_types", [])
            if isinstance(item, dict)
        ]
        normalized_domains[key] = normalized

    payload["domains"] = normalized_domains
    payload["default_domains"] = [
        _domain_key(name)
        for name in payload.get("default_domains", ["core"])
        if _domain_key(name) in normalized_domains
    ] or ["core"]
    payload["auto_min_keyword_matches"] = int(payload.get("auto_min_keyword_matches", 3))
    return payload


def configured_domain_names(domain_config: dict[str, Any], requested: str) -> list[str]:
    domains = domain_config["domains"]
    names = [_domain_key(name) for name in str(requested or "").split(",") if _domain_key(name)]
    if not names:
        names = list(domain_config.get("default_domains", ["core"]))

    active: list[str] = []
    for name in [*domain_config.get("default_domains", ["core"]), *names]:
        if name in domains and name not in active:
            active.append(name)
    return active


def classify_document_domains(
    domain_config: dict[str, Any],
    text: str,
) -> tuple[list[str], dict[str, Any]]:
    normalized_text = str(text or "").lower()
    domains = domain_config["domains"]
    active = list(domain_config.get("default_domains", ["core"]))
    scores: dict[str, int] = {}
    matched_keywords: dict[str, list[str]] = {}

    for name, spec in domains.items():
        if name in active:
            continue
        keywords = spec.get("keywords", [])
        matches = sorted(keyword for keyword in keywords if keyword in normalized_text)
        scores[name] = len(matches)
        matched_keywords[name] = matches[:20]
        threshold = int(spec.get("min_keyword_matches", domain_config["auto_min_keyword_matches"]))
        if len(matches) >= threshold:
            active.append(name)

    document_type = next((name for name in active if name not in domain_config["default_domains"]), "generic")
    return active, {
        "mode": "auto",
        "document_type": document_type,
        "scores": scores,
        "matched_keywords": matched_keywords,
    }


def domain_ontology_entries(
    domain_names: str,
    path: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    domain_config = load_domain_config(path)
    requested = configured_domain_names(domain_config, domain_names)

    vertices: list[dict] = []
    edges: list[dict] = []
    active: list[str] = []
    for domain in requested:
        spec = domain_config["domains"].get(domain)
        if not spec:
            continue
        active.append(domain)
        vertices.extend(dict(item) for item in spec.get("vertex_types", []))
        edges.extend(dict(item) for item in spec.get("edge_types", []))

    return vertices, edges, active
