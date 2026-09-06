PYTHON ?= python

.PHONY: help install verify lint test demo figures package

help:
	@printf '%s\n' \
	  'install  Install the lightweight project interface' \
	  'verify   Check workflow, result, and provenance paths' \
	  'lint     Check supported Python code and formatting' \
	  'test     Run the test suite' \
	  'demo     Run the synthetic reference-assisted transfer example' \
	  'figures  Rebuild figures when the required local inputs are present' \
	  'package  Build the deterministic reproducibility archive'

install:
	$(PYTHON) -m pip install -e .

verify:
	PYTHONPATH=src $(PYTHON) -m tessera_forest_structure verify

lint:
	$(PYTHON) -m ruff check src tests tools workflows/figure_generation workflows/sensitivity_analysis
	$(PYTHON) -m ruff format --check src tests tools workflows/figure_generation workflows/sensitivity_analysis

test:
	$(PYTHON) -m pytest -q

demo:
	PYTHONPATH=src $(PYTHON) -m tessera_forest_structure demo

figures:
	$(PYTHON) workflows/figure_generation/make_figures.py

package: verify lint test
	PYTHONPATH=src $(PYTHON) tools/build_reproducibility_package.py
