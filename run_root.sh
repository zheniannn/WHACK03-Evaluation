#!/usr/bin/env bash
# Run the full WHACK03 discrimination pipeline against a given data root.
# Usage: run_root.sh /abs/path/to/data_root
set -uo pipefail
export WHACK_DATA_ROOT="$1"
export PYTHONUNBUFFERED=1
cd /home/ian/working/WHACK/WHACK03-Evaluation
PY=/home/ian/working/WHACK/.venv/bin/python
echo "######## WHACK03 @ WHACK_DATA_ROOT=$WHACK_DATA_ROOT ########"
for s in 10_tracking.py 11_M-of-N.py 12_SPRT.py 13_IMM.py 14_GBM.py 15_GRU.py 16_VAE.py 17_Fusion.py 18_Latent-SDE.py 19_Evaluation.py; do
  echo "==== [$(date +%H:%M:%S)] $s ===="
  "$PY" "scripts/$s"; rc=$?
  if [ $rc -ne 0 ]; then echo "!!! $s FAILED rc=$rc"; exit $rc; fi
done
echo "######## DONE ########"
