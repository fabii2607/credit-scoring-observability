"""Lista os alvos do Makefile com a descrição após '##' (usado por `make help`)."""

import re
from pathlib import Path

for line in Path("Makefile").read_text(encoding="utf-8").splitlines():
    match = re.match(r"^([a-zA-Z_-]+):.*?## (.*)$", line)
    if match:
        print(f"  {match.group(1):<20} {match.group(2)}")
