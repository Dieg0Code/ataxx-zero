param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("up", "down", "new", "current", "heads", "history", "stamp")]
    [string]$Command,

    [Parameter(Position = 1)]
    [string]$Arg1,

    [Parameter(Position = 2)]
    [string]$Arg2
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot
try {
    switch ($Command) {
        "up" {
            $target = if ($Arg1) { $Arg1 } else { "head" }
            uv run alembic upgrade $target
        }
        "down" {
            $target = if ($Arg1) { $Arg1 } else { "-1" }
            uv run alembic downgrade $target
        }
        "new" {
            if (-not $Arg1) {
                throw "Usage: .\scripts\db.ps1 new ""message"""
            }
            uv run alembic revision --autogenerate -m $Arg1
        }
        "current" {
            uv run alembic current
        }
        "heads" {
            uv run alembic heads
        }
        "history" {
            uv run alembic history
        }
        "stamp" {
            $target = if ($Arg1) { $Arg1 } else { "head" }
            uv run alembic stamp $target
        }
    }
}
finally {
    Pop-Location
}
