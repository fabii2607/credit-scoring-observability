import subprocess
import sys
from pathlib import Path

from credit_scoring_observability.logger import setup_logging, get_logger
from credit_scoring_observability.tracing import setup_tracing

# Inicializa o logging (com gravação em arquivo para o Alloy) e o tracing
setup_logging(log_file=Path("logs/pipeline.jsonl"))
setup_tracing()

logger = get_logger(__name__)

def main():
    # Simulando os meses M1 a M7 da Etapa 2
    batches = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07"]
    
    sucesso = 0
    bloqueados = 0
    falhas_erro = 0
    
    logger.info("iniciando_monitoramento_em_lote", total_batches=len(batches))

    for batch in batches:
        logger.info("processando_lote", batch_id=batch)
        
        # Executa o run_batch como um processo isolado
        result = subprocess.run(
            [sys.executable, "-m", "credit_scoring_observability.run_batch", "--batch", batch],
            capture_output=False # Deixa os logs aparecerem no terminal
        )
        
        if result.returncode == 0:
            logger.info("lote_processado_com_sucesso", batch_id=batch)
            sucesso += 1
        elif result.returncode == 2:
            logger.warning("lote_bloqueado_pelo_contrato", batch_id=batch)
            bloqueados += 1
        else:
            logger.error("falha_inesperada_no_lote", batch_id=batch, exit_code=result.returncode)
            falhas_erro += 1

    logger.info(
        "monitoramento_concluido", 
        sucesso=sucesso, 
        bloqueados=bloqueados, 
        falhas=falhas_erro
    )

if __name__ == "__main__":
    main()