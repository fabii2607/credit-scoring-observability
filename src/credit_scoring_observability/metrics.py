import os
from pathlib import Path
from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway, write_to_textfile

class MetricsPublisher:
    """
    Publica métricas do pipeline para o Prometheus via Pushgateway ou arquivo local.
    """
    
    # Dicionário de classe para reaproveitar os Gauges instanciados,
    # já que o Prometheus não permite registrar a mesma métrica duas vezes.
    _gauges: dict[str, Gauge] = {}

    def __init__(self, batch_id: str, stage: str | None = None):
        self.batch_id = batch_id
        self.stage = stage
        self.registry = CollectorRegistry()
        
        # O agrupamento principal no Pushgateway será por batch_id e stage
        self.grouping_key = {"batch_id": self.batch_id}
        if self.stage:
            self.grouping_key["stage"] = self.stage

    def set(self, name: str, value: float, **labels) -> None:
        """Cria um Gauge sob demanda e define seu valor."""
        # Combina os labels padrões com os passados no método
        all_labels = {"batch_id": self.batch_id}
        if self.stage:
            all_labels["stage"] = self.stage
        all_labels.update(labels)

        # Se o Gauge ainda não existe, criamos ele
        if name not in self.__class__._gauges:
            label_names = list(all_labels.keys())
            self.__class__._gauges[name] = Gauge(
                name=name,
                documentation=f"Métrica {name}",
                labelnames=label_names,
                registry=self.registry
            )
        else:
            # Se já existe, precisamos garantir que ele está atrelado ao registro atual
            self.registry.register(self.__class__._gauges[name])
            
        # Seta o valor da métrica com os labels informados
        self.__class__._gauges[name].labels(**all_labels).set(value)

    def publish(self) -> None:
        """
        Envia as métricas via pushadd (POST) para o Pushgateway.
        Se PUSHGATEWAY_URL estiver vazio, salva em .prom localmente.
        """
        pushgateway_url = os.environ.get("PUSHGATEWAY_URL", "").strip()

        if pushgateway_url:
            try:
                # Usa pushadd para não sobrescrever o grupo inteiro (armadilha evitada)
                pushadd_to_gateway(
                    gateway=pushgateway_url,
                    job="credit_scoring_pipeline",
                    registry=self.registry,
                    grouping_key=self.grouping_key
                )
            except Exception as e:
                # Falha de rede vira apenas um aviso, não trava o pipeline
                import logging
                logging.warning(f"Falha ao enviar métricas para o Pushgateway: {e}")
        else:
            # Fallback para ambiente offline: gravar em reports/metrics/
            metrics_dir = Path("reports/metrics")
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            stage_suffix = f"_{self.stage}" if self.stage else ""
            file_path = metrics_dir / f"{self.batch_id}{stage_suffix}.prom"
            
            write_to_textfile(str(file_path), self.registry)