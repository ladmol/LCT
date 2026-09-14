.PHONY: help backend-dev backend-prod lint lint-fix typecheck check

help:
	@echo "backend-dev   - run backend with auto-reload (fastapi dev)"
	@echo "backend-prod  - run backend in production mode (fastapi run)"
	@echo "lint          - ruff check on backend/ and ml/"
	@echo "lint-fix      - ruff check and fix on backend/ and ml/"
	@echo "typecheck     - pyrefly on backend/ and ml/"
	@echo "check         - lint + typecheck"

backend-dev:
	$(MAKE) -C backend dev

backend-prod:
	$(MAKE) -C backend prod

lint:
	$(MAKE) -C backend lint
	$(MAKE) -C ml lint

lint-fix:
	$(MAKE) -C backend lint-fix
	$(MAKE) -C ml lint-fix

typecheck:
	$(MAKE) -C backend typecheck
	$(MAKE) -C ml typecheck

check: lint typecheck
