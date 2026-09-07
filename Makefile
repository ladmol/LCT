.PHONY: help backend-dev backend-prod

help:
	@echo "backend-dev   - run backend with auto-reload (fastapi dev)"
	@echo "backend-prod  - run backend in production mode (fastapi run)"

backend-dev:
	cd backend && uv run fastapi dev app/main.py

backend-prod:
	cd backend && uv run fastapi run app/main.py
