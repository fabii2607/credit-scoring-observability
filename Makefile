.PHONY: help setup lint lint-fix test test-unit test-cov prepare train validate demo-contract simulate monitor fairness causal purge-quarantine clean

PKG := credit_scoring_observability

help: ## Show this help message
	@uv run python scripts/make_help.py

# ── Setup e qualidade ────────────────────────────────────────────────────────
setup: ## Install dependencies, pre-commit hooks and create .env
	uv sync
	uv run pre-commit install
	@uv run python -c "import shutil, pathlib; p=pathlib.Path('.env'); p.exists() or (shutil.copy('.env.example', '.env'), print('[OK] .env criado a partir do .env.example'))"

lint: ## Run ruff linter and formatter check
	uv run ruff check .
	uv run ruff format --check .

lint-fix: ## Auto-fix lint issues
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run all tests (synthetic data, no Kaggle needed)
	uv run pytest

test-unit: ## Run only the fast unit tests
	uv run pytest -m "not integration"

test-cov: ## Run tests with coverage report
	uv run pytest --cov

# ── Etapa 1: dados, contrato e modelo ────────────────────────────────────────
prepare: ## Split Reference / Production pool from data/raw (pseudonymized, no leakage)
	uv run python -m $(PKG).prepare

train: ## Train the V2 model and save it with its decision threshold in models/
	uv run python -m $(PKG).train

validate: ## Validate one batch file through the data contract (BATCH=path; exit 2 = blocked)
	uv run python -m $(PKG).validate --batch $(BATCH)

demo-contract: ## Push a corrupted batch through the data contract (expected to FAIL, exit 2)
	-uv run python -m $(PKG).validate --demo-corrupted

# ── Etapa 2: simulação e drift ───────────────────────────────────────────────
simulate: ## Generate the monthly production batches M1–M7 from the pool
	@echo "TODO (Etapa 2): ver docs/handoff/etapa-2-drift.md"

# ── Etapa 3: observabilidade ─────────────────────────────────────────────────
monitor: ## Run the batch pipeline for every simulated month
	@echo "TODO (Etapa 3): ver docs/handoff/etapa-3-observabilidade.md"

# ── Etapa 4: governança ──────────────────────────────────────────────────────
fairness: ## Fairness by region and threshold mitigation report
	@echo "TODO (Etapa 4): ver docs/handoff/etapa-4-governanca.md"

causal: ## Causal attribution of the drift (interventions by variable group)
	@echo "TODO (Etapa 4): ver docs/handoff/etapa-4-governanca.md"

purge-quarantine: ## Delete quarantined batches older than the retention period (LGPD)
	@echo "TODO (Etapa 4): ver docs/handoff/etapa-4-governanca.md"

clean: ## Remove caches
	uv run python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree('.pytest_cache', ignore_errors=True); shutil.rmtree('.ruff_cache', ignore_errors=True)"
