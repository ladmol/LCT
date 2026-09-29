.PHONY: help lint lint-fix format typecheck test compile check

help:
	@echo "lint       - ruff check"
	@echo "lint-fix   - ruff check and fix"
	@echo "format     - verify formatting of the Re-ID code"
	@echo "typecheck  - pyrefly basic checks for the Re-ID code"
	@echo "test       - unit and smoke tests"
	@echo "compile    - compile Python modules"
	@echo "check      - all checks used before commit"

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

format:
	uv run ruff format --check attributes eval reid scripts tests

typecheck:
	uv run pyrefly check -p basic --project-excludes detection/** --project-excludes tools/**

test:
	uv run pytest -q

compile:
	uv run python -m compileall -q reid eval scripts tests

check: lint format typecheck test compile
