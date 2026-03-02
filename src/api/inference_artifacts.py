from __future__ import annotations

import importlib
import os
from functools import lru_cache
from urllib.parse import parse_qs, unquote, urlparse


def _normalize_local_artifact_path(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"Unsupported artifact URI host: {uri}")
        path = unquote(parsed.path)
        if path == "":
            raise ValueError(f"Invalid artifact URI (empty path): {uri}")
        return path
    if parsed.scheme == "":
        return uri
    raise ValueError(
        f"Unsupported artifact URI scheme '{parsed.scheme}'. Use local paths, file://, or hf:// URIs."
    )


def _parse_hf_uri(
    uri: str,
    *,
    default_repo_id: str | None = None,
    default_revision: str | None = None,
) -> tuple[str, str, str | None]:
    parsed = urlparse(uri)
    if parsed.scheme != "hf":
        raise ValueError(f"Invalid HF URI: {uri}")

    # Expected format: hf://<owner>/<repo>/<path/to/file>[?revision=<rev>]
    payload = f"{parsed.netloc}{parsed.path}".lstrip("/")
    segments = [seg for seg in payload.split("/") if seg]
    if len(segments) >= 3:
        repo_id = f"{segments[0]}/{segments[1]}"
        filename = "/".join(segments[2:])
    elif len(segments) >= 1 and default_repo_id:
        repo_id = default_repo_id
        filename = "/".join(segments)
    else:
        raise ValueError(
            f"Invalid HF URI '{uri}'. Expected hf://<owner>/<repo>/<artifact_path>."
        )

    if filename == "":
        raise ValueError(f"Invalid HF URI '{uri}': missing artifact path.")

    query = parse_qs(parsed.query)
    revision = query.get("revision", [None])[0] or default_revision
    return repo_id, filename, revision


@lru_cache(maxsize=128)
def _download_hf_artifact(
    *,
    repo_id: str,
    filename: str,
    revision: str | None,
) -> str:
    try:
        hf_module = importlib.import_module("huggingface_hub")
    except ModuleNotFoundError as exc:
        raise ValueError(
            "huggingface-hub is required for hf:// artifacts. "
            "Install it in the API runtime dependencies."
        ) from exc

    hf_hub_download = getattr(hf_module, "hf_hub_download", None)
    if hf_hub_download is None:
        raise ValueError("huggingface_hub.hf_hub_download is unavailable in this runtime.")

    token = os.getenv("HF_TOKEN")
    return str(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token if token else None,
        )
    )


def resolve_artifact_uri(
    uri: str | None,
    *,
    default_repo_id: str | None = None,
    default_revision: str | None = None,
) -> str | None:
    if uri is None:
        return None
    cleaned = uri.strip()
    if cleaned == "":
        return None

    parsed = urlparse(cleaned)
    if parsed.scheme in {"", "file"}:
        return _normalize_local_artifact_path(cleaned)
    if parsed.scheme == "hf":
        repo_id, filename, revision = _parse_hf_uri(
            cleaned,
            default_repo_id=default_repo_id,
            default_revision=default_revision,
        )
        return _download_hf_artifact(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )
    raise ValueError(
        f"Unsupported artifact URI scheme '{parsed.scheme}'. Use local paths, file://, or hf:// URIs."
    )
