VENV := .venv

PROJECT := service
TESTS := tests
DEV_MODELS := dev_models

IMAGE_NAME := reco_service
CONTAINER_NAME := reco_service

# Prepare

.venv:
	poetry install --no-root
	poetry check

setup: .venv


# Clean

clean:
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf $(VENV)


# Format

isort_fix: .venv
	poetry run isort $(PROJECT) $(TESTS) $(DEV_MODELS)


black_fix:
	poetry run black $(PROJECT) $(TESTS) $(DEV_MODELS)

format: isort_fix black_fix


# Lint

isort: .venv
	poetry run isort --check $(PROJECT) $(TESTS) $(DEV_MODELS)

.black:
	poetry run black --check --diff $(PROJECT) $(TESTS) $(DEV_MODELS)

flake: .venv
	poetry run flake8 $(PROJECT) $(TESTS) $(DEV_MODELS)

mypy: .venv
	poetry run mypy $(PROJECT) $(TESTS) $(DEV_MODELS)

pylint: .venv
	poetry run pylint $(PROJECT) $(TESTS) $(DEV_MODELS)

lint: isort flake mypy pylint


# Test

.pytest:
	poetry run pytest $(TESTS)

test: .venv .pytest


# Docker

build:
	docker build . -t $(IMAGE_NAME)

run: build
	docker run -p 8080:8080 --name $(CONTAINER_NAME) $(IMAGE_NAME)

# All

all: setup format lint test run

.DEFAULT_GOAL = all
