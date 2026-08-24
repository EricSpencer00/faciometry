# Vitruve
#
# `make install` does one thing that looks superstitious and is not: it ad-hoc
# codesigns every compiled extension in the virtualenv. On macOS, XProtect deep
# scans an unsigned .dylib the first time it is loaded, and a fresh install of
# opencv, mediapipe and numpy is enough of them that the first import hangs for
# minutes with no output. Signing them at install time moves that cost to a
# place where it is visible and pays it once.

PY := .venv/bin/python
VITRUVE := .venv/bin/vitruve
EXTRAS := permissive,api,dev

.DEFAULT_GOAL := help
.PHONY: help install test lint evals serve demo clean

help:  ## what each target does
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## create the venv, install the permissive stack, sign the extensions
	uv venv --python 3.11
	uv pip install --python $(PY) -e '.[$(EXTRAS)]'
	@echo "signing compiled extensions so XProtect does not stall the first import"
	@find .venv \( -name "*.so" -o -name "*.dylib" \) -print0 \
		| xargs -0 -P 8 -n 20 codesign -s - -f 2>/dev/null || true
	@$(VITRUVE) doctor

test:  ## the whole suite
	$(PY) -m pytest

lint:  ## ruff and mypy
	$(PY) -m ruff check src tests
	$(PY) -m mypy

evals:  ## the validation arms in evals/
	@if [ -f evals/harness/run.py ]; then \
		$(PY) evals/harness/run.py $(ARMS); \
	else \
		echo "evals/harness/run.py is not present in this checkout."; \
		echo "The design of the arms is in docs/superpowers/specs, section 4."; \
		exit 1; \
	fi

.venv/.installed: pyproject.toml  ## marker: the package itself is importable from the venv
	@test -x $(PY) || uv venv --python 3.11
	uv pip install --python $(PY) -e '.[$(EXTRAS)]'
	@echo "signing compiled extensions so XProtect does not stall the first import"
	@find .venv \( -name "*.so" -o -name "*.dylib" \) -print0 \
		| xargs -0 -P 8 -n 20 codesign -s - -f 2>/dev/null || true
	@touch $@

serve: .venv/.installed  ## the local API and web UI on 127.0.0.1:8731
	$(PY) -m vitruve serve

demo: .venv/.installed  ## what the tool says before any photograph is taken
	$(PY) -m vitruve doctor
	@echo
	$(PY) -m vitruve catalogue
	@echo
	$(PY) -m vitruve licenses --tier permissive

clean:  ## remove caches and build output
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	rm -rf src/*.egg-info out runs
	find . -path ./.venv -prune -o -name "__pycache__" -type d -print0 | xargs -0 rm -rf

app: .venv/.installed  ## build the signed macOS application bundle and dmg
	packaging/macos/build_app.sh
