.PHONY: help dev prod lint lint-fix typecheck check

help:
	@echo "dev        - run backend with auto-reload (fastapi dev)"
	@echo "prod       - run backend in production mode (fastapi run)"
	@echo "lint       - ruff check"
	@echo "lint-fix   - ruff check and fix"
	@echo "typecheck  - ty + pyrefly"
	@echo "check      - lint + typecheck"

dev:
	uv run fastapi dev app/main.py

prod:
	uv run fastapi run app/main.py

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

typecheck:
	uv run ty check .
	uv run pyrefly check .

check: lint typecheck
