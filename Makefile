.PHONY: test demo check benchmark neural-v2-smoke

test:
	python -m pytest -q

demo:
	python -m universa_recurrent.cli demo --trace full

check:
	python -m compileall -q src tests examples experiments
	python -m pytest -q

benchmark:
	python experiments/benchmark.py --output runs/benchmark.json

neural-v2-smoke:
	python -m universa_recurrent.cli neural-v2-train --device cpu \
		--train-size 48 --calibration-size 24 --epochs 1 --batch-size 16 \
		--hidden-dim 12 --candidate-embedding-dim 4 --steps 2 \
		--output /tmp/universa-recurrent-v2-smoke.pt
	python -m universa_recurrent.cli neural-v2-demo --device cpu \
		--checkpoint /tmp/universa-recurrent-v2-smoke.pt \
		--output /tmp/universa-recurrent-v2-smoke.json
	python -m universa_recurrent.cli neural-v2-verify \
		/tmp/universa-recurrent-v2-smoke.json \
		--checkpoint /tmp/universa-recurrent-v2-smoke.pt
