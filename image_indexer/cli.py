"""idx CLI — visual memory search.

Commands:
  status   Show database statistics
  search   Search indexed images (semantic, lexical, structured)
  index    Index a directory of images via RunPod

Design:
  - JSON output for agents (--json flag)
  - Exit codes: 0=success, 1=user error, 2=system error, 3=partial failure
  - Progress to stderr, data to stdout
  - No interactive prompts ever
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import click

DEFAULT_DB = Path.home() / ".local" / "share" / "image-indexer" / "index.db"

# Exit codes
EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_SYSTEM_ERROR = 2
EXIT_PARTIAL = 3


def common_options(f):
    """Decorator that adds --json, --verbose, --db to any click command."""
    @click.option("--json", "output_json", is_flag=True, help="JSON output (for agents)")
    @click.option("--verbose", is_flag=True, help="Show progress to stderr")
    @click.option("--db", "db_path", type=click.Path(), default=str(DEFAULT_DB),
                  help="Database path")
    @functools.wraps(f)
    def wrapper(*args, output_json, verbose, db_path, **kwargs):
        ctx = click.get_current_context()
        ctx.ensure_object(dict)
        ctx.obj.update(json=output_json, verbose=verbose, db_path=Path(db_path))
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return f(*args, **kwargs)
    return wrapper


def emit(ctx, data, summary_fn=None):
    """Emit output: JSON for agents, human-readable for terminals."""
    if ctx.obj.get("json"):
        click.echo(json.dumps(data, indent=2, default=str))
    elif summary_fn:
        summary_fn(data)
    else:
        click.echo(json.dumps(data, indent=2, default=str))


def log(ctx, msg):
    """Progress message to stderr (only when --verbose)."""
    if ctx.obj.get("verbose"):
        click.echo(msg, err=True)


@click.group()
def main():
    """idx — visual memory search.

    Find anything with pixels by what you remember, not what you named it.
    """
    pass


# ── status ──────────────────────────────────────────────────────────────────

@main.command()
@common_options
def status():
    """Show database statistics."""
    ctx = click.get_current_context()
    from image_indexer.db import connect

    try:
        db = connect(ctx.obj["db_path"])
    except Exception as e:
        click.echo(f"Error: cannot open database: {e}", err=True)
        sys.exit(EXIT_SYSTEM_ERROR)

    image_count = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    last = db.execute("SELECT MAX(updated_at) FROM images").fetchone()[0]

    data = {
        "database": str(ctx.obj["db_path"]),
        "images": image_count,
        "last_indexed": last,
    }

    def summary(d):
        click.echo(f"Database: {d['database']}")
        click.echo(f"Images:   {d['images']}")
        if d["last_indexed"]:
            click.echo(f"Last:     {d['last_indexed']}")

    emit(ctx, data, summary)
    sys.exit(EXIT_OK)


# ── search ──────────────────────────────────────────────────────────────────

@main.command()
@click.argument("query")
@click.option("--semantic", is_flag=True, help="Vector similarity (SigLIP2)")
@click.option("--lexical", is_flag=True, help="Full-text over captions + OCR")
@click.option("--structured", is_flag=True, help="SQL WHERE on EXIF/metadata")
@click.option("--limit", "-n", type=int, default=10, help="Max results")
@common_options
def search(query, semantic, lexical, structured, limit):
    """Search indexed images.

    QUERY is a text search term, or a SQL WHERE clause with --structured.
    """
    ctx = click.get_current_context()
    from image_indexer.db import connect

    if not (semantic or lexical or structured):
        click.echo(
            "Error: pick --semantic, --lexical, or --structured", err=True
        )
        sys.exit(EXIT_USER_ERROR)

    try:
        db = connect(ctx.obj["db_path"])
    except Exception as e:
        click.echo(f"Error: cannot open database: {e}", err=True)
        sys.exit(EXIT_SYSTEM_ERROR)

    results = []

    if semantic:
        from typing import cast
        from image_indexer.text_embed import TextEmbedder
        embedder = TextEmbedder()
        log(ctx, "Embedding query via SigLIP2 (local)...")
        query_vec = cast(list[float], embedder.embed(query))
        from image_indexer.db import search_semantic
        results.extend(search_semantic(db, query_vec, k=limit))

    if lexical:
        from image_indexer.db import search_lexical
        results.extend(search_lexical(db, query, k=limit))

    if structured:
        from image_indexer.db import search_structured
        results.extend(search_structured(db, query))

    # Deduplicate by id when multiple modes overlap.
    seen = set()
    unique = []
    for r in results:
        key = r.get("image_id") or r.get("id")
        if key not in seen:
            seen.add(key)
            unique.append(r)
    results = unique[:limit]

    def summary(rows):
        if not rows:
            click.echo("No results.")
            return
        for row in rows:
            path = row.get("path", "?")
            desc = row.get("description", "")[:80]
            meta = ""
            if row.get("distance") is not None:
                meta = f"  dist={row['distance']:.4f}"
            elif row.get("score") is not None:
                meta = f"  score={row['score']:.3f}"
            click.echo(f"{path}{meta}")
            if desc:
                click.echo(f"  {desc}")

    emit(ctx, results, summary)
    sys.exit(EXIT_OK)


# ── index ───────────────────────────────────────────────────────────────────

@main.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.option("--endpoint-id", envvar="RUNPOD_ENDPOINT_ID",
              help="RunPod endpoint ID")
@click.option("--api-key", envvar="RUNPOD_API_KEY",
              help="RunPod API key")
@click.option("--dry-run", is_flag=True,
              help="Preprocess only, skip RunPod")
@common_options
def index(directory, endpoint_id, api_key, dry_run):
    """Index a directory of images.

    Idempotent: re-running skips already-indexed files (SHA-256 dedup).
    """
    ctx = click.get_current_context()
    from image_indexer.db import connect, upsert_image
    from image_indexer.preprocess import preprocess, scan_directory, sha256_bytes

    if not dry_run and not endpoint_id:
        click.echo("Error: --endpoint-id or RUNPOD_ENDPOINT_ID required", err=True)
        sys.exit(EXIT_USER_ERROR)
    if not dry_run and not api_key:
        click.echo("Error: --api-key or RUNPOD_API_KEY required", err=True)
        sys.exit(EXIT_USER_ERROR)

    dir_path = Path(directory)
    log(ctx, f"Scanning {dir_path}...")
    image_paths = scan_directory(dir_path)
    log(ctx, f"Found {len(image_paths)} images")

    try:
        db = connect(ctx.obj["db_path"])
    except Exception as e:
        click.echo(f"Error: cannot open database: {e}", err=True)
        sys.exit(EXIT_SYSTEM_ERROR)

    client = None
    if not dry_run:
        from image_indexer.client import RunPodClient
        client = RunPodClient(endpoint_id=endpoint_id, api_key=api_key)

    stats = {"indexed": 0, "skipped": 0, "failed": 0}

    for path in image_paths:
        log(ctx, f"  {path.name}...")

        # SHA-256 dedup against database.
        try:
            digest = sha256_bytes(path.read_bytes())
            existing = db.execute(
                "SELECT id FROM images WHERE sha256 = ?", (digest,)
            ).fetchone()
            if existing:
                log(ctx, "    already indexed")
                stats["skipped"] += 1
                continue
        except OSError:
            stats["failed"] += 1
            continue

        prep = preprocess(path)
        if prep.skipped:
            log(ctx, f"    skipped: {prep.skip_reason}")
            stats["skipped"] += 1
            continue

        if dry_run:
            stats["indexed"] += 1
            continue

        try:
            result = client.run(prep.jpeg_bytes, task="all")
        except Exception as e:
            log(ctx, f"    inference failed: {e}")
            stats["failed"] += 1
            continue

        meta = {
            "path": str(prep.path),
            "sha256": prep.sha256,
            "file_size": prep.file_size,
            "format": prep.disk_format,
            "width": prep.orig_width,
            "height": prep.orig_height,
            "description": result.get("description"),
            "model_caption": "Qwen/Qwen3-VL-4B-Instruct",
            "model_embed": "google/siglip2-so400m-patch16-384",
        }
        try:
            upsert_image(db, meta, embedding=result.get("embedding"))
            stats["indexed"] += 1
        except Exception as e:
            log(ctx, f"    db error: {e}")
            stats["failed"] += 1

    stats["database"] = str(ctx.obj["db_path"])

    def summary(s):
        click.echo(f"Indexed:  {s['indexed']}")
        click.echo(f"Skipped:  {s['skipped']}")
        click.echo(f"Failed:   {s['failed']}")
        click.echo(f"Database: {s['database']}")

    emit(ctx, stats, summary)
    sys.exit(EXIT_PARTIAL if stats["failed"] > 0 else EXIT_OK)


if __name__ == "__main__":
    main()
