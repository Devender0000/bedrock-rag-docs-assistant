PYTHON ?= python3

.PHONY: install lint test synth deploy destroy ingest eval-check eval ui

install:
	$(PYTHON) -m pip install -r requirements.txt -r infra/requirements.txt

lint:
	ruff check .

test:
	$(PYTHON) -m pytest -q

synth:
	cd infra && cdk synth

deploy:
	cd infra && cdk deploy

destroy:
	cd infra && cdk destroy

ingest:
	$(PYTHON) -m ingestion.ingest

eval-check:
	$(PYTHON) -m eval.run_eval --check-questions

eval:
	$(PYTHON) -m eval.run_eval --label baseline --judge

ui:
	streamlit run app/streamlit_app.py
