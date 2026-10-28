#!/bin/sh
# Run ONCE in the server's existing root terminal if automatic upload is wanted.
# Adds this public key to cjm only. Does not alter root's restricted SSH gate.
set -eu
test "$(id -u)" = 0
test "$(hostname -s)" = admin1
test "$(getent passwd cjm | cut -d: -f6)" = /home/cjm
directory=/home/cjm/.ssh
target=$directory/authorized_keys
test ! -L "$directory"
test ! -L "$target"
group=$(id -gn cjm)
install -d -m 700 -o cjm -g "$group" "$directory"
if [ -f "$target" ]; then
  cp -p "$target" "$target.before_ggca_$(date +%Y%m%d_%H%M%S)"
fi
key='restrict ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINzM/poXGegPIN3FQYXA3TkldgrL2faI64YDinL4dV8g codex-cjm-ggca-20261027'
if ! grep -Fqx "$key" "$target" 2>/dev/null; then
  printf '\n%s\n' "$key" >> "$target"
fi
chown cjm:"$group" "$target"
chmod 600 "$target"
if command -v restorecon >/dev/null 2>&1; then restorecon "$directory" "$target"; fi
printf '%s\n' 'CJM_GGCA_SSH_READY; root restricted access remains unchanged'
