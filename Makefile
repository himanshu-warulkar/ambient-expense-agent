.PHONY: install playground run test lint

install:
	agents-cli install

playground:
	PORT=8080 uv run python -m expense_agent.fast_api_app

run:
	PORT=8080 uv run python -m expense_agent.fast_api_app

test:
	uv run pytest

lint:
	agents-cli lint
