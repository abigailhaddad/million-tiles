#!/usr/bin/env bash
# Fetch the public-domain US Census data the equal-population mosaics need (~14 MB).
# Re-runnable: skips files that already exist. No API key required.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

CENPOP=https://www2.census.gov/geo/docs/reference/cenpop2020
fetch () {  # url  dest
  if [ -s "$2" ]; then echo "have   $2"; else echo "get -> $2"; curl -fsS "$1" -o "$2"; fi
}

fetch "$CENPOP/blkgrp/CenPop2020_Mean_BG.txt"  data/cenpop2020_bg.txt
fetch "$CENPOP/tract/CenPop2020_Mean_TR.txt"   data/cenpop2020_tract.txt
fetch "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv" \
      data/county_pop_2023.csv

echo "done. (State boundary GeoJSON is fetched on demand into data/cache/ on first render.)"
