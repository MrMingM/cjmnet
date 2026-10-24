ROOT=/data/cjm/datasets/logs/uecp_gspr_v2xr_v1
ZIP=/home/cjm/uecp_gspr_resume_fix_v2.zip
EVIDENCE="$ROOT/diagnostics/controlled_resume_20261020T083508_121163Z_3625228/RESULTS.json"
if [ -f "$ZIP" ] && [ -f "$EVIDENCE" ]; then
    LOG=$(mktemp "$ROOT/diagnostics/resume_v2_launcher.XXXXXX")
    nohup /home/cjm/miniconda3/envs/opencood/bin/python -B "$ZIP" run \
      --output "$ROOT" --evidence "$EVIDENCE" --mode all \
      > "$LOG" 2>&1 < /dev/null &
    echo "PID=$! LOG=$LOG"
    tail -f "$LOG"
else
    echo "请检查ZIP已上传，且原始RESULTS.json仍在指定路径"
fi
