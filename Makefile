.PHONY: install clean lint sync-deps help lock-deps check-deps ci run

# Default target
help:
	@echo "Available targets:"
	@echo ""
	@echo "General:"
	@echo "  install      - Install saber-agent in development mode"
	@echo "  clean        - Remove build artifacts and cache files"
	@echo "  lint         - Run linter and fix issues with ruff"
	@echo "  sync-deps    - Sync uv dependencies and lock file"
	@echo "  run.         - Run saber agent"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Dependency Management:"
	@echo "  lock-deps    - Update dependency lock file"
	@echo "  check-deps   - Check if dependencies are up to date"
	@echo "  ci           - Run the same checks as GitHub CI (lint, format check, tests)"

# Install saber-agent in development mode
install:
	uv pip install -e .

# Clean build artifacts and cache files
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .pyright/
	rm -rf .ruff_cache/

# Run linter and fix issues
lint:
	uv run ruff check --fix .
	uv run ruff format .

# Mirror the GitHub Actions CI locally
ci:
	uv sync --group dev
	uv run ruff check
	uv run ruff format --check

# Sync uv dependencies
sync-deps:
	uv sync

# Run saber agent
run:
	@echo "⚾ Starting Saber Agent..."
	uv run saber-agent

# Dependency Management Commands

# Update dependency lock file
lock-deps:
	@echo "🔒 Updating dependency lock file..."
	uv lock
	@echo "✅ Lock file updated (uv.lock)"

# Check if dependencies are up to date
check-deps:
	@echo "🔍 Checking dependency status..."
	@if uv lock --check > /dev/null 2>&1; then \
		echo "✅ Dependencies are up to date"; \
	else \
		echo "⚠️  Dependencies need updating. Run 'make lock-deps'"; \
	fi
