from __future__ import annotations

import shlex


def _tokenize_args(raw_args: str) -> list[str]:
    try:
        return shlex.split(raw_args)
    except ValueError:
        # Fallback keeps launch robust when user-provided args contain malformed quotes.
        return raw_args.split()


def _has_flag(tokens: list[str], flag: str) -> bool:
    if flag in tokens:
        return True
    return any(token.startswith(f"{flag}=") for token in tokens)


def build_pod_env(
    *,
    hf_token: str | None,
    hf_repo_id: str,
    hf_run_id: str,
) -> list[dict[str, str]]:
    token = (hf_token or "").strip()
    return [
        {"key": "PYTHONUNBUFFERED", "value": "1"},
        {"key": "HF_TOKEN", "value": token},
        {"key": "HF_REPO_ID", "value": hf_repo_id},
        {"key": "HF_RUN_ID", "value": hf_run_id},
    ]


def build_train_start_command(
    *,
    repository: str,
    git_ref: str,
    train_args: str,
    hf_repo_id: str,
    hf_run_id: str,
) -> str:
    if hf_repo_id.strip() == "":
        raise ValueError("hf_repo_id is required for RunPod training uploads.")
    if hf_run_id.strip() == "":
        raise ValueError("hf_run_id is required for RunPod training uploads.")

    # We pin to a git ref to make every run reproducible from the exact commit.
    safe_repo = shlex.quote(repository)
    safe_ref = shlex.quote(git_ref)
    train_tokens = _tokenize_args(train_args.strip())
    if not _has_flag(train_tokens, "--hf"):
        train_tokens.append("--hf")
    if not _has_flag(train_tokens, "--save-every"):
        # Frequent snapshots reduce checkpoint loss when spot pods are interrupted.
        train_tokens.extend(["--save-every", "1"])
    safe_train_args = " ".join(shlex.quote(token) for token in train_tokens)
    safe_hf_repo = shlex.quote(hf_repo_id)
    safe_hf_run = shlex.quote(hf_run_id)
    return (
        "bash -lc 'set -e; "
        "cd /workspace; "
        f"if [ ! -d ataxx-zero ]; then git clone https://github.com/{safe_repo} ataxx-zero; fi; "
        "cd ataxx-zero; "
        "git fetch --all --tags; "
        f"git checkout {safe_ref}; "
        "curl -LsSf https://astral.sh/uv/install.sh | sh; "
        "export PATH=$HOME/.local/bin:$PATH; "
        "test -n \"${HF_TOKEN:-}\" || (echo \"Missing HF_TOKEN in pod env\" && exit 2); "
        "uv sync --frozen --group train --group export; "
        f"uv run python train.py {safe_train_args} "
        f"--hf-repo-id {safe_hf_repo} --hf-run-id {safe_hf_run}'"
    )


def pod_finished(*, desired_status: str, runtime_status: str) -> bool:
    desired = desired_status.strip().upper()
    runtime = runtime_status.strip().upper()
    return desired in {"STOPPED", "TERMINATED"} or runtime in {"EXITED", "FAILED"}
