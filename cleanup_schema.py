#!/usr/bin/env python3
"""
Drop Oracle objects owned by a user for a given graph-swarm schema prefix.

The script connects as db_user, removes prefixed SQL property graphs first,
then drops prefixed materialized views, views, standalone indexes, and tables.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from dataclasses import dataclass

import oracledb


DEFAULT_SCHEMA_PREFIX = "SW1_"
DEFAULT_DB_USER = "graphuser"
DEFAULT_DB_PASSWORD = "Welcome12345"
DEFAULT_DB_CONNECTION = "localhost:1521/FREEPDB1"

MISSING_OBJECT_ERROR_CODES = {
    942,    # ORA-00942: table or view does not exist
    1418,   # ORA-01418: specified index does not exist
    4043,   # ORA-04043: object does not exist
    12003,  # ORA-12003: materialized view does not exist
    42421,  # ORA-42421: property graph does not exist
}

VECTOR_AUXILIARY_TABLE_ERROR_CODES = {
    51903,  # ORA-51903: Cannot drop auxiliary tables for vector indexes
}

CONSTRAINT_INDEX_ERROR_CODES = {
    2429,  # ORA-02429: cannot drop index used for unique/primary key
}


@dataclass(frozen=True)
class CleanupTargets:
    property_graphs: list[str]
    materialized_views: list[str]
    views: list[str]
    tables: list[str]
    indexes: list[str]

    def has_anything_to_drop(self) -> bool:
        return any(
            (
                self.property_graphs,
                self.materialized_views,
                self.views,
                self.tables,
                self.indexes,
            )
        )


def validate_identifier(value: str, label: str) -> str:
    """Validate an unquoted Oracle identifier or identifier prefix."""
    value = value.strip().upper()
    if not value:
        raise ValueError(f"{label} cannot be empty")
    if len(value) > 128:
        raise ValueError(f"{label} must be 128 characters or fewer")
    if not value[0].isalpha():
        raise ValueError(f"{label} must start with a letter")
    if not all(char.isalnum() or char == "_" for char in value):
        raise ValueError(
            f"{label} can only contain letters, numbers, and underscores"
        )
    return value


def quote_identifier(name: str) -> str:
    """Quote a dictionary-returned object name for DDL."""
    return '"' + name.replace('"', '""') + '"'


def like_pattern_for_prefix(prefix: str) -> str:
    """Build an escaped LIKE pattern so underscores remain literal."""
    return (
        prefix.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
        + "%"
    )


def like_pattern_containing_prefix(prefix: str) -> str:
    """Build an escaped LIKE pattern for names containing the prefix."""
    return "%" + like_pattern_for_prefix(prefix)


def oracle_error_code(exc: oracledb.DatabaseError) -> int | None:
    if not exc.args:
        return None
    return getattr(exc.args[0], "code", None)


def is_missing_object_error(exc: oracledb.DatabaseError) -> bool:
    code = oracle_error_code(exc)
    if code in MISSING_OBJECT_ERROR_CODES:
        return True
    text = str(exc).lower()
    return "does not exist" in text or "not exist" in text


def fetch_names(
    cursor: oracledb.Cursor,
    sql: str,
    prefix_pattern: str,
) -> list[str]:
    cursor.execute(sql, {"prefix_pattern": prefix_pattern})
    return [row[0] for row in cursor.fetchall()]


def fetch_names_if_view_exists(
    cursor: oracledb.Cursor,
    sql: str,
    prefix_pattern: str,
) -> list[str]:
    try:
        return fetch_names(cursor, sql, prefix_pattern)
    except oracledb.DatabaseError as exc:
        if oracle_error_code(exc) == 942:
            return []
        raise


def unique_ordered(names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name not in seen:
            unique.append(name)
            seen.add(name)
    return unique


def discover_property_graphs(
    cursor: oracledb.Cursor,
    prefix_pattern: str,
) -> list[str]:
    """
    Discover SQL property graph names.

    USER_PROPERTY_GRAPHS is the expected Oracle 23ai dictionary view. The
    USER_OBJECTS fallback keeps discovery working across dictionary variants.
    """
    names: list[str] = []
    names.extend(
        fetch_names_if_view_exists(
            cursor,
            """
            SELECT graph_name
            FROM user_property_graphs
            WHERE graph_name LIKE :prefix_pattern ESCAPE '\\'
            ORDER BY graph_name
            """,
            prefix_pattern,
        )
    )
    names.extend(
        fetch_names(
            cursor,
            """
            SELECT object_name
            FROM user_objects
            WHERE UPPER(object_type) LIKE '%PROPERTY GRAPH%'
              AND object_name LIKE :prefix_pattern ESCAPE '\\'
            ORDER BY object_name
            """,
            prefix_pattern,
        )
    )
    return unique_ordered(names)


def discover_indexes(
    cursor: oracledb.Cursor,
    containing_prefix_pattern: str,
) -> list[str]:
    """Find user-visible standalone indexes that should be dropped before tables."""
    names: list[str] = []
    names.extend(
        fetch_names(
            cursor,
            """
            SELECT i.index_name
            FROM user_indexes i
            WHERE i.index_name LIKE :prefix_pattern ESCAPE '\\'
              AND i.index_type <> 'LOB'
              AND i.index_name NOT LIKE 'SYS\\_%' ESCAPE '\\'
              AND i.index_name NOT LIKE 'VECTOR$%'
              AND NOT EXISTS (
                  SELECT 1
                  FROM user_constraints c
                  WHERE c.index_name = i.index_name
                    AND c.constraint_type IN ('P', 'U')
              )
            ORDER BY i.index_name
            """,
            containing_prefix_pattern,
        )
    )
    names.extend(
        fetch_names(
            cursor,
            """
            SELECT i.index_name
            FROM user_indexes i
            WHERE i.index_type = 'VECTOR'
              AND i.table_name LIKE :prefix_pattern ESCAPE '\\'
              AND NOT EXISTS (
                  SELECT 1
                  FROM user_constraints c
                  WHERE c.index_name = i.index_name
                    AND c.constraint_type IN ('P', 'U')
              )
            ORDER BY i.index_name
            """,
            containing_prefix_pattern,
        )
    )
    return unique_ordered(names)


def discover_targets(
    connection: oracledb.Connection,
    schema_prefix: str,
) -> CleanupTargets:
    prefix_pattern = like_pattern_for_prefix(schema_prefix)
    containing_prefix_pattern = like_pattern_containing_prefix(schema_prefix)
    with connection.cursor() as cursor:
        property_graphs = discover_property_graphs(cursor, prefix_pattern)
        materialized_views = fetch_names(
            cursor,
            """
            SELECT mview_name
            FROM user_mviews
            WHERE mview_name LIKE :prefix_pattern ESCAPE '\\'
            ORDER BY mview_name
            """,
            prefix_pattern,
        )
        views = fetch_names(
            cursor,
            """
            SELECT view_name
            FROM user_views
            WHERE view_name LIKE :prefix_pattern ESCAPE '\\'
            ORDER BY view_name
            """,
            prefix_pattern,
        )
        tables = fetch_names(
            cursor,
            """
            SELECT table_name
            FROM user_tables
            WHERE table_name LIKE :prefix_pattern ESCAPE '\\'
              AND nested = 'NO'
              AND secondary = 'N'
              AND table_name NOT LIKE 'VECTOR$%'
            ORDER BY table_name
            """,
            containing_prefix_pattern,
        )
        indexes = discover_indexes(
            cursor,
            containing_prefix_pattern,
        )

    return CleanupTargets(
        property_graphs=property_graphs,
        materialized_views=materialized_views,
        views=views,
        tables=tables,
        indexes=indexes,
    )


def drop_objects(
    connection: oracledb.Connection,
    object_type_label: str,
    ddl_template: str,
    names: Iterable[str],
    dry_run: bool,
) -> int:
    dropped = 0
    names = list(names)
    if not names:
        return dropped

    with connection.cursor() as cursor:
        for name in names:
            ddl = ddl_template.format(name=quote_identifier(name))
            if dry_run:
                print(f"DRY RUN: {ddl}")
                continue
            try:
                print(f"Dropping {object_type_label}: {name}")
                cursor.execute(ddl)
                dropped += 1
            except oracledb.DatabaseError as exc:
                if is_missing_object_error(exc):
                    print(f"Already gone: {object_type_label} {name}")
                    continue
                if (
                    object_type_label == "table"
                    and oracle_error_code(exc) in VECTOR_AUXILIARY_TABLE_ERROR_CODES
                ):
                    print(
                        "Skipping Oracle vector-index auxiliary table "
                        f"{name}; drop the owning vector index instead."
                    )
                    continue
                if (
                    object_type_label == "index"
                    and oracle_error_code(exc) in CONSTRAINT_INDEX_ERROR_CODES
                ):
                    print(
                        f"Skipping constraint-backed index {name}; "
                        "drop the owning table or constraint instead."
                    )
                    continue
                raise
    return dropped


def drop_remaining_indexes(
    connection: oracledb.Connection,
    schema_prefix: str,
    dry_run: bool,
) -> int:
    """Drop standalone prefixed indexes left after table/view removal."""
    containing_prefix_pattern = like_pattern_containing_prefix(schema_prefix)
    with connection.cursor() as cursor:
        indexes = discover_indexes(
            cursor,
            containing_prefix_pattern,
        )
    return drop_objects(connection, "index", "DROP INDEX {name}", indexes, dry_run)


def print_targets(targets: CleanupTargets) -> None:
    groups = [
        ("property graphs", targets.property_graphs),
        ("materialized views", targets.materialized_views),
        ("views", targets.views),
        ("tables", targets.tables),
        ("indexes", targets.indexes),
    ]
    for label, names in groups:
        print(f"{label}: {len(names)}")
        for name in names:
            print(f"  - {name}")


def confirm_or_exit(schema_prefix: str, targets: CleanupTargets, yes: bool) -> None:
    if yes or not targets.has_anything_to_drop():
        return

    expected = f"DROP {schema_prefix}"
    print()
    print(f"This will permanently drop objects owned by the connected user.")
    print(f"Type {expected!r} to continue:")
    answer = input("> ").strip().upper()
    if answer != expected:
        print("Aborted.")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop all tables owned by db_user whose names contain "
            "schema_prefix, plus SQL property-graph related objects whose "
            "names start with schema_prefix."
        )
    )
    parser.add_argument(
        "--schema-prefix",
        default=DEFAULT_SCHEMA_PREFIX,
        help=f"Object-name prefix to clean up (default: {DEFAULT_SCHEMA_PREFIX})",
    )
    parser.add_argument(
        "--db-user",
        default=DEFAULT_DB_USER,
        help=f"Oracle database user/schema (default: {DEFAULT_DB_USER})",
    )
    parser.add_argument(
        "--db-password",
        default=DEFAULT_DB_PASSWORD,
        help="Oracle database password",
    )
    parser.add_argument(
        "--db-connection",
        default=DEFAULT_DB_CONNECTION,
        help=f"Oracle DSN (default: {DEFAULT_DB_CONNECTION})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show matching objects and DDL without dropping anything",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema_prefix = validate_identifier(args.schema_prefix, "schema_prefix")
    db_user = validate_identifier(args.db_user, "db_user")

    print(f"Connecting to {args.db_connection} as {db_user}")
    with oracledb.connect(
        user=db_user,
        password=args.db_password,
        dsn=args.db_connection,
    ) as connection:
        targets = discover_targets(connection, schema_prefix)
        print()
        print(f"Objects matching prefix {schema_prefix!r}:")
        print_targets(targets)

        if not targets.has_anything_to_drop():
            print("Nothing to drop.")
            return 0

        if args.dry_run:
            print()
            print("Dry run requested; no objects will be dropped.")
        else:
            confirm_or_exit(schema_prefix, targets, args.yes)

        drop_counts = {
            "property_graphs": drop_objects(
                connection,
                "property graph",
                "DROP PROPERTY GRAPH {name}",
                targets.property_graphs,
                args.dry_run,
            ),
            "materialized_views": drop_objects(
                connection,
                "materialized view",
                "DROP MATERIALIZED VIEW {name}",
                reversed(targets.materialized_views),
                args.dry_run,
            ),
            "views": drop_objects(
                connection,
                "view",
                "DROP VIEW {name}",
                reversed(targets.views),
                args.dry_run,
            ),
            "indexes": drop_objects(
                connection,
                "index",
                "DROP INDEX {name}",
                targets.indexes,
                args.dry_run,
            ),
            "tables": drop_objects(
                connection,
                "table",
                "DROP TABLE {name} CASCADE CONSTRAINTS PURGE",
                reversed(targets.tables),
                args.dry_run,
            ),
        }

        drop_counts["remaining_indexes"] = (
            0
            if args.dry_run
            else drop_remaining_indexes(connection, schema_prefix, args.dry_run)
        )

        if not args.dry_run:
            connection.commit()

    print()
    print("Cleanup summary:")
    for label, count in drop_counts.items():
        print(f"  {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
