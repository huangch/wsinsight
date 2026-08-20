#!/usr/bin/env bash
# Match the in-container "user" to the UID/GID that actually owns the mounted
# /workspace, at *run* time, so the baked build-time uid never has to match the
# host. Order of precedence for the target ids:
#   1. explicit HOST_UID / HOST_GID environment variables, else
#   2. the owner of the mounted /workspace, else
#   3. 1000:1000.
#
# The container normally starts as root so it can remap; the final command is
# then executed as the target user via setpriv (util-linux; no extra package).
# If the container was started with `--user` (already non-root), remapping is
# neither possible nor needed, so we just exec the command as-is.
#
# Kept byte-for-byte in sync across wsinsight, sptxinsight, hplot and
# wsinsight-train so every image behaves identically at run time.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    # Started with `docker run --user ...`: honor it verbatim.
    exec "$@"
fi

TARGET_UID="${HOST_UID:-$(stat -c %u /workspace 2>/dev/null || echo 1000)}"
TARGET_GID="${HOST_GID:-$(stat -c %g /workspace 2>/dev/null || echo 1000)}"

# Remap the pre-created "user" account (and its group) to the target ids.
# -o allows a non-unique id in case it collides with an existing account.
if [ "$(id -g user 2>/dev/null || echo -1)" != "$TARGET_GID" ]; then
    groupmod -o -g "$TARGET_GID" user 2>/dev/null || true
fi
if [ "$(id -u user 2>/dev/null || echo -1)" != "$TARGET_UID" ]; then
    usermod -o -u "$TARGET_UID" -g "$TARGET_GID" user 2>/dev/null || true
fi

# Keep the home directory and image-owned caches usable by the remapped user.
chown "$TARGET_UID:$TARGET_GID" /home/user 2>/dev/null || true
# hf-cache (when present) is a possibly large named volume; only recurse when
# its top-level owner doesn't already match, so normal restarts stay fast.
if [ -d /app/hf-cache ] && \
   [ "$(stat -c %u /app/hf-cache 2>/dev/null || echo -1)" != "$TARGET_UID" ]; then
    chown -R "$TARGET_UID:$TARGET_GID" /app/hf-cache 2>/dev/null || true
fi

export HOME=/home/user
# Drop root and run the requested command as the target user with its groups.
exec setpriv --reuid "$TARGET_UID" --regid "$TARGET_GID" --init-groups "$@"
