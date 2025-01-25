#!/usr/bin/env bash
set -x
set -e

python3 depth_anything_v2/generate.py val
python3 depth_anything_v2/generate.py test
python3 marigold_v1-0/generate.py val
python3 marigold_v1-0/generate.py test
