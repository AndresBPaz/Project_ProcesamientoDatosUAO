#!/usr/bin/env bash
# Arranca la interfaz de PEGASUS dentro del codespace.
# El puerto 8501 se reenvia solo; para compartir la URL con la clase hay que
# ponerlo en Public (panel PORTS, clic derecho -> Port Visibility -> Public,
# o:  gh codespace ports visibility 8501:public -c "$CODESPACE_NAME")
set -euo pipefail

exec streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false