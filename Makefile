.PHONY: test demo check benchmark neural-smoke

test:
	python -m pytest -q

demo:
	python -m universa_recurrent.cli demo --trace full

check:
	python -m compileall -q src tests examples experiments
	python -m pytest -q

benchmark:
	python experiments/benchmark.py --output runs/classical-benchmark.json

neural-smoke:
	python -m universa_recurrent.cli neural-train --device cpu --train-size 64 --val-size 32 --epochs 1 --batch-size 32 --hidden-dim 16 --steps 2 --output /tmp/universa-neural-smoke.pt
	python -m universa_recurrent.cli neural-demo --device cpu --checkpoint /tmp/universa-neural-smoke.pt --max-steps 2 --fixed --output /tmp/universa-neural-smoke.json
	python -m universa_recurrent.cli neural-verify /tmp/universa-neural-smoke.json --checkpoint /tmp/universa-neural-smoke.pt
