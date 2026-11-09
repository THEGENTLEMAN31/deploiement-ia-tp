SHELL := /bin/bash
COMPOSE := docker compose

.PHONY: build shell prepare notebooks report all clean-results clean

build:
	$(COMPOSE) build

shell:
	$(COMPOSE) run --rm tp bash

prepare:
	$(COMPOSE) run --rm tp python -m src.prepare

notebooks:
	$(COMPOSE) run --rm tp bash scripts/execute_notebooks.sh

report:
	$(COMPOSE) run --rm tp python -m src.make_plots
	$(COMPOSE) run --rm tp python -m src.make_report

all: prepare notebooks report

clean-results:
	rm -f results/*.json results/*.csv results/*.npy report/figures/*.png

clean: clean-results
	rm -rf .cache .home
