# bench/Makefile - everything you need on a laptop with no GPU.
#
#   make venv      create .venv with uv and install the dependencies
#   make test      run the whole test suite (this must pass before committing)
#   make mock      start the mock server on :8001 for manual poking
#
PY := .venv/bin/python
UV := uv

.PHONY: help venv test mock clean

help:
	@sed -n '2,6p' Makefile | sed 's/^# \{0,1\}//'

venv:
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python .venv -e '.[test]'

$(PY):
	$(MAKE) venv

test: $(PY)
	$(PY) -m pytest tests

mock: $(PY)
	$(PY) mock_server.py --port 8001 --ttft-ms 150 --tpot-ms 8

clean:
	rm -rf results/local results/test .pytest_cache **/__pycache__
