"""Credit scoring com contratos de dados, detecção de drift e observabilidade.

Privacy by default: as bibliotecas abaixo enviam estatísticas de uso para os
fornecedores quando nada é configurado. O opt-out fica aqui, antes de qualquer
import delas, para valer em todo ponto de entrada (CLI, testes, notebooks).
"""

import os
import sys

for _var, _value in {
    "GX_ANALYTICS_ENABLED": "False",  # Great Expectations
    "NML_DISABLE_USAGE_LOGGING": "1",  # NannyML
    "MLFLOW_DISABLE_TELEMETRY": "true",  # MLflow
    "MLFLOW_DISABLE_AGENT_HINT": "1",  # MLflow (aviso no import)
    "EVIDENTLY_DISABLE_TELEMETRY": "1",  # Evidently
    "DO_NOT_TRACK": "1",  # convenção genérica
}.items():
    os.environ.setdefault(_var, _value)

# No Windows a saída redirecionada usa cp1252 e quebra com os emojis que o MLflow
# imprime. UTF-8 com substituição evita isso.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")
