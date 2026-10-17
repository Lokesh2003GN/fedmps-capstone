#!/usr/bin/env bash
# CCTV benchmark on the AWS GPU instance: download ChokePoint, extract it,
# run both benchmarks on CUDA. Run from the project root, inside the project
# virtualenv (HOW_TO_RUN.md section 2.1: requirements + insightface
# onnxruntime-gpu).
#
#   bash cctv_benchmark/run_on_aws.sh [data_dir]
#
# data_dir defaults to ~/chokepoint. About 12 GB to download. Downloads resume
# where they stopped and finished archives are skipped, so it is safe to re-run.
#
# Layout: the cropped-face archives and the full-frame archives both contain
# folders named like P1E_S1_C1, so they are extracted into separate trees:
#   <data_dir>/groundtruth/   eye annotations
#   <data_dir>/crops/         P1E.tar.xz, P1L.tar.xz, P2E.tar.xz, P2L.tar.xz
#   <data_dir>/frames/        P1E_S1.tar.xz ... P2L_S4.tar.xz
set -euo pipefail

DATA="${1:-$HOME/chokepoint}"
BASE="https://zenodo.org/api/records/815657/files"
CROPS="P1E P1L P2E P2L"
FRAMES="P1E_S1 P1E_S2 P1E_S3 P1E_S4 P1L_S1 P1L_S2 P1L_S3 P1L_S4 \
P2E_S1 P2E_S2 P2E_S3 P2E_S4 P2L_S1 P2L_S2 P2L_S3 P2L_S4"
# P2E_S5 / P2L_S5 (crowded, continuous occlusion) are outside the official
# G1/G2 protocol and are not downloaded.

fetch() {   # fetch <archive name> <extract into>
    local f="$1" dest="$2" archive="$DATA/$1.tar.xz"
    [ -f "$DATA/.extracted_$f" ] && return 0
    echo "==> $f"
    for i in $(seq 1 20); do
        curl -sSL -C - --speed-limit 20000 --speed-time 60 \
             -o "$archive" "$BASE/$f.tar.xz/content" && break
        echo "    retry $i"; sleep 5
    done
    mkdir -p "$dest"
    tar -xJf "$archive" -C "$dest"
    # The full-frame archives hold one more archive per camera
    # (P1E_S1.tar.xz -> P1E_S1_C1.tar.xz ...); unpack those too.
    for inner in "$dest"/"$f"_C*.tar.xz; do
        [ -f "$inner" ] && tar -xJf "$inner" -C "$dest" && rm -f "$inner"
    done
    touch "$DATA/.extracted_$f" && rm -f "$archive"
}

mkdir -p "$DATA"
fetch groundtruth "$DATA"
# Zenodo throttles each connection to ~100-300 KB/s but not the total, so
# every archive downloads at once: ~1.5-2 h for 12 GB instead of ~17 h.
for f in $CROPS;  do fetch "$f" "$DATA/crops"  > "$DATA/fetch_$f.log" 2>&1 & done
for f in $FRAMES; do fetch "$f" "$DATA/frames" > "$DATA/fetch_$f.log" 2>&1 & done
wait
missing=""
for f in $CROPS $FRAMES; do [ -f "$DATA/.extracted_$f" ] || missing="$missing $f"; done
if [ -n "$missing" ]; then
    echo "not downloaded:$missing - re-run this script to resume"; exit 1
fi
GT="$(dirname "$(find "$DATA/groundtruth" -name 'P1E_S1_C1.xml' | head -1)")"

python - <<'EOF'
import onnxruntime as ort, torch
print("onnxruntime providers:", ort.get_available_providers())
print("torch cuda:", torch.cuda.is_available())
EOF

# Part A - recognition on real CCTV faces: official crops, then full frames.
python cctv_benchmark/chokepoint_recognition.py --root "$DATA/crops" --groundtruth "$GT" \
    --source crops --device cuda --out cctv_chokepoint_recognition_crops.json
python cctv_benchmark/chokepoint_recognition.py --root "$DATA/frames" --groundtruth "$GT" \
    --source frames --device cuda --out cctv_chokepoint_recognition_frames.json

# Part B - the live pipeline end to end on every protocol sequence.
python cctv_benchmark/chokepoint_video.py --root "$DATA/frames" --groundtruth "$GT" \
    --device cuda --out cctv_chokepoint_video.json

echo "done - results in results/cctv_chokepoint_*.json"
