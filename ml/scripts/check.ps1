$ErrorActionPreference = "Stop"

uv run ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run ruff format --check attributes eval reid scripts tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run pyrefly check -p basic --project-excludes "detection/**" --project-excludes "tools/**"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run python -m compileall -q reid eval scripts tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "ML checks passed."
