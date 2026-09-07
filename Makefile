.PHONY: test demo check benchmark

test:
	python -m pytest -q

demo:
	python -m universa_recurrent.cli demo --trace full

check:
	python -m compileall -q src tests examples experiments
	python -m pytest -q

benchmark:
	python experiments/benchmark.py --output runs/benchmark.json
