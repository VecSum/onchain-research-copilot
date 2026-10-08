PY ?= python3

.PHONY: test eval demo ingest clean

test:
	$(PY) -m unittest discover -s tests -v

eval:
	$(PY) -m eval.run_eval --report output/eval-report.md

demo:
	$(PY) -m onchain_copilot research --offline --config config.example.toml \
		--pool 0xdemo000000000000000000000000000000000001 \
		--question "is the pool being pushed by single trades" \
		--out output/demo-report.html

ingest:
	$(PY) -m onchain_copilot --config config.toml ingest

clean:
	rm -rf output
