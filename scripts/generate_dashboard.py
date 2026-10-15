import json
from pathlib import Path

def create_dashboard():
    dashboard = {
        "title": "Observabilidade - Credit Scoring",
        "tags": ["mlops", "fase4", "pipeline"],
        "timezone": "browser",
        "refresh": "5s",
        "panels": [
            # ── LINHA 1: Qualidade e Negócio (y=0) ──
            {
                "title": "Status do Pipeline",
                "type": "stat",
                "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0},
                "targets": [{"expr": "pipeline_stage_status", "legendFormat": "{{batch_id}} - {{stage}}"}],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "red", "value": None},
                                {"color": "green", "value": 1}
                            ]
                        },
                        "mappings": [
                            {"type": "value", "options": {"0": {"text": "FALHA"}, "1": {"text": "SUCESSO"}}}
                        ]
                    }
                }
            },
            {
                "title": "Lotes em Quarentena",
                "type": "stat",
                "gridPos": {"h": 8, "w": 8, "x": 8, "y": 0},
                "targets": [{"expr": "batch_quarantined", "legendFormat": "{{batch_id}}"}],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "red", "value": 1} # Vermelho se entrar em quarentena
                            ]
                        }
                    }
                }
            },
            {
                "title": "Violações de Contrato (Qualidade de Dados)",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 8, "x": 16, "y": 0},
                "targets": [{"expr": "contract_violations", "legendFormat": "{{rule}} - {{severity}}"}]
            },
            
            # ── LINHA 2: Infraestrutura (y=8) ──
            {
                "title": "Duração do Estágio (s)",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 8, "x": 0, "y": 8},
                "targets": [{"expr": "pipeline_stage_duration_seconds", "legendFormat": "{{batch_id}} - {{stage}}"}]
            },
            {
                "title": "Uso de CPU (s)",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 8, "x": 8, "y": 8},
                "targets": [{"expr": "stage_cpu_seconds", "legendFormat": "{{batch_id}} - {{stage}}"}]
            },
            {
                "title": "Pico de Memória (Bytes)",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 8, "x": 16, "y": 8},
                "targets": [{"expr": "stage_memory_peak_bytes", "legendFormat": "{{batch_id}} - {{stage}}"}]
            },

            # ── LINHA 3: Drift e Performance - Preparado para a Etapa 2 (y=16) ──
            {
                "title": "Data Drift (PSI) - Requer Etapa 2",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                "targets": [{"expr": "model_psi_score", "legendFormat": "{{feature}}"}]
            },
            {
                "title": "Performance do Modelo (KS) - Requer Etapa 2",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                "targets": [{"expr": "model_ks_score", "legendFormat": "Score"}]
            },

            # ── LINHA 4: Logs centralizados do Loki (y=24) ──
            {
                "title": "Logs Estruturados do Pipeline",
                "type": "logs",
                "datasource": "Loki", # Garante que vai buscar à Data Source do Loki
                "gridPos": {"h": 12, "w": 24, "x": 0, "y": 24},
                "targets": [{"expr": "{batch_id=~\"2026-.*\"}"}], # Busca todos os logs dos lotes 2026
                "options": {
                    "showTime": True,
                    "showLabels": False,
                    "showCommonLabels": False,
                    "wrapLogMessage": True
                }
            }
        ]
    }
    
    out_dir = Path("monitoring/grafana")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = out_dir / "dashboard.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2)
        
    print(f"Dashboard gerado com sucesso em: {file_path}")

if __name__ == "__main__":
    create_dashboard()