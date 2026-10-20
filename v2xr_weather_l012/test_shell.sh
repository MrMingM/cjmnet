#!/bin/sh
# Pure synthetic launcher test: replace Python with a shell shim; no scientific runtime.
set -eu
tmp=$1
launcher=$2
mkdir -p "$tmp/path with spaces"
shim="$tmp/path with spaces/python shim"
capture="$tmp/args.txt"
cat > "$shim" <<'SHIM'
#!/bin/sh
case "$1" in -B) shift ;; esac
case "$1" in *check_static.py) exit 0 ;; esac
printf '%s\n' "$@" > "$WEATHER_SHELL_CAPTURE"
[ "${HIP_VISIBLE_DEVICES+x}" != x ] || exit 91
[ "${CUDA_VISIBLE_DEVICES+x}" != x ] || exit 92
[ "$ROCR_VISIBLE_DEVICES" = 3 ] || exit 93
exit 37
SHIM
chmod +x "$shim"
set +e
env PYTHON="$shim" GPU=3 MODE=all RESUME=1 \
    TMPDIR="$tmp/tmp space" OUTPUT_ROOT="$tmp/output with spaces" TRAIN_ROOT="$tmp/train with spaces" \
    HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=6 WEATHER_SHELL_CAPTURE="$capture" \
    sh "$launcher"
code=$?
set -e
[ "$code" = 37 ] || { echo "Exit propagation failed: $code" >&2; exit 1; }
found_output=0
found_train=0
found_resume=0
while IFS= read -r argument; do
    [ "$argument" != "$tmp/output with spaces" ] || found_output=1
    [ "$argument" != "$tmp/train with spaces" ] || found_train=1
    [ "$argument" != --resume ] || found_resume=1
done < "$capture"
[ "$found_output$found_train$found_resume" = 111 ]
echo 'POSIX SHELL PASS: LF, quoted paths, env GPU filtering, RESUME and exit propagation.'
