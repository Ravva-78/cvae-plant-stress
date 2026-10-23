.PHONY: help install test lint demo demo-fast webapp plots clean

PYTHON ?= python

help:
	@echo "CVAE Plant Stress Detector — Developer Commands"
	@echo "================================================"
	@echo "make install    : Install runtime and developer dependencies"
	@echo "make test       : Run comprehensive pytest test suite with coverage"
	@echo "make lint       : Run static code analysis and style checks"
	@echo "make demo       : Run 1-command zero-setup CLI demonstration"
	@echo "make demo-fast  : Run fast CI test run verification pass"
	@echo "make webapp     : Launch interactive Flask web application"
	@echo "make plots      : Regenerate all evaluation and benchmark plots"
	@echo "make clean      : Remove Python bytecode, test caches, and coverage artifacts"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install pytest pytest-cov flake8 black

test:
	$(PYTHON) -m pytest -v --cov=utils --cov=webapp --cov-report=term-missing

lint:
	flake8 utils webapp demo.py tests --max-line-length=127 --statistics

demo:
	$(PYTHON) demo.py

demo-fast:
	$(PYTHON) demo.py --test-run

webapp:
	$(PYTHON) webapp/app.py

plots:
	$(PYTHON) scripts/generate_plots.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[cod]" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov coverage.xml dist build *.egg-info 2>/dev/null || true
