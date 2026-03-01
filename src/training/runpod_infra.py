from __future__ import annotations

import shlex


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
    # We pin to a git ref to make every run reproducible from the exact commit.
    safe_repo = shlex.quote(repository)
    safe_ref = shlex.quote(git_ref)
    safe_train_args = train_args.strip()
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
        "uv sync --frozen --group train --group export; "
        f"uv run python train.py {safe_train_args} "
        f"--hf-repo-id {safe_hf_repo} --hf-run-id {safe_hf_run}'"
    )


def pod_finished(*, desired_status: str, runtime_status: str) -> bool:
    desired = desired_status.strip().upper()
    runtime = runtime_status.strip().upper()
    return desired in {"STOPPED", "TERMINATED"} or runtime in {"EXITED", "FAILED"}
