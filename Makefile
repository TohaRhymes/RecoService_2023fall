VENV := .venv

PROJECT := service
TESTS := tests
DEV_MODELS := dev_models
MODELS := models

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
	poetry run isort $(PROJECT) $(TESTS) $(DEV_MODELS) $(MODELS)


black_fix:
	poetry run black $(PROJECT) $(TESTS) $(DEV_MODELS) $(MODELS)

format: isort_fix black_fix


# Lint

isort: .venv
	poetry run isort --check $(PROJECT) $(TESTS) $(DEV_MODELS) $(MODELS)

.black:
	poetry run black --check --diff $(PROJECT) $(TESTS) $(DEV_MODELS) $(MODELS)

flake: .venv
	poetry run flake8 $(PROJECT) $(TESTS) $(DEV_MODELS) $(MODELS)

mypy: .venv
	poetry run mypy $(PROJECT) $(TESTS) $(DEV_MODELS) $(MODELS)

pylint: .venv
	poetry run pylint $(PROJECT) $(TESTS)

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
