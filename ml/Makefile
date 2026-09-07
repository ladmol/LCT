.PHONY: help lint lint-fix typecheck check

help:
	@echo "lint       - ruff check"
	@echo "lint-fix   - ruff check and fix"
	@echo "typecheck  - ty + pyrefly"
	@echo "check      - lint + typecheck"

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

typecheck:
	uv run ty check .
	uv run pyrefly check .

check: lint typecheck
