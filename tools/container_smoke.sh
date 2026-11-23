#!/usr/bin/env bash
# Kindi: prove a release image end to end, in the image itself.
#
#   bash tools/container_smoke.sh [--image TAG] [--volume NAME] [--port P] [--name C]
#                                 [--wait SECONDS] [--cpu] [--dry-run] [--keep]
#
# The image must already be built and the named volume must hold a pack at /data/pack:
#
#   docker build -t kindi:0.7.0-fork -f docker/Dockerfile .
#   docker volume create kindi-smoke-data
#   docker cp ~/.kindi/packs/bonsai-2b/. kindi-smoke-data:/data/pack   (see docs/DOCKER.md)
#
# What it proves, with the image's own Python, CuPy and server packages:
#   1. `kindi selftest` one-shot: reads the pack manifest, builds the tokenizer and the engine,
#      then calls /health and one chat completion through the ASGI route. No port is bound and
#      no container is left behind.
#   2. `kindi selftest --url http://127.0.0.1:8093` inside the serving container once /health
#      answers: the real Uvicorn route and the chat route, with the key the container carries.
#
# The server container is removed on exit unless --keep. --dry-run prints every command and
# changes nothing. DOCKER overrides the docker binary; KINDI_IMAGE, KINDI_VOLUME, KINDI_PORT,
# KINDI_SMOKE_NAME and KINDI_SMOKE_WAIT set the defaults below.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
DOCKER=${DOCKER:-docker}
VERSION=$(sed -n 's/^VERSION *= *"\(.*\)"/\1/p' "$HERE/kindi/__init__.py")
IMAGE=${KINDI_IMAGE:-kindi:${VERSION:-0.7.0-fork}}
VOLUME=${KINDI_VOLUME:-kindi-smoke-data}
PORT=${KINDI_PORT:-18093}
NAME=${KINDI_SMOKE_NAME:-kindi-smoke}
WAIT=${KINDI_SMOKE_WAIT:-300}
KEY=kindi-smoke
GPU=1 DRY=0 KEEP=0 STARTED=0

usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; }
die() { echo "container_smoke: $*" >&2; exit 1; }
say() { echo "container_smoke: $*"; }
show() { printf 'container_smoke:   %s\n' "$*"; }
cleanup() {
    if [ "$STARTED" = 1 ] && [ "$KEEP" = 0 ]; then
        "$DOCKER" rm -f "$NAME" >/dev/null 2>&1 || true
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        --image) IMAGE=$2; shift 2 ;;
        --volume) VOLUME=$2; shift 2 ;;
        --port) PORT=$2; shift 2 ;;
        --name) NAME=$2; shift 2 ;;
        --wait) WAIT=$2; shift 2 ;;
        --cpu) GPU=0; shift ;;
        --dry-run) DRY=1; shift ;;
        --keep) KEEP=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown option $1" ;;
    esac
done
case "$WAIT" in ''|*[!0-9]*) die "--wait must be a whole number of seconds" ;; esac

# Git Bash rewrites /data in -v arguments; the 2026-11-20 container smoke needed this too.
case "$(uname -s 2>/dev/null || echo unknown)" in MINGW*|MSYS*|CYGWIN*) export MSYS_NO_PATHCONV=1 ;; esac

GPU_ARGS=""
[ "$GPU" = 1 ] && GPU_ARGS="--gpus all"
SMOKE_ENV="-e KINDI_PACK=/data/pack -e KINDI_STATE_DIR=/data/state -e KINDI_CTX=4096 -e KINDI_API_KEY=$KEY"

if [ "$DRY" = 1 ]; then
    show "$DOCKER run --rm -v $VOLUME:/data --entrypoint sh $IMAGE -c 'test -d /data/pack && ... test -w /data/state'"
    show "$DOCKER run --rm $GPU_ARGS -v $VOLUME:/data $SMOKE_ENV --entrypoint python3 $IMAGE -m kindi selftest"
    show "$DOCKER run -d --name $NAME $GPU_ARGS -p 127.0.0.1:$PORT:8093 -v $VOLUME:/data $SMOKE_ENV -e KINDI_HOST=0.0.0.0 $IMAGE"
    show "$DOCKER exec $NAME python3 -c \"... urlopen('http://127.0.0.1:8093/health', timeout=3)\"    # until it answers, up to ${WAIT}s"
    show "$DOCKER exec $NAME python3 -m kindi selftest --url http://127.0.0.1:8093"
    exit 0
fi

command -v "$DOCKER" >/dev/null 2>&1 || die "no ${DOCKER} on PATH"
"$DOCKER" image inspect "$IMAGE" >/dev/null 2>&1 || die "no image $IMAGE (build it first: docker build -t $IMAGE -f docker/Dockerfile .)"
if ! "$DOCKER" run --rm -v "$VOLUME:/data" --entrypoint sh "$IMAGE"         -c 'test -d /data/pack && mkdir -p /data/state && test -w /data/state'; then
    die "volume $VOLUME needs a pack at /data/pack and write access for uid 1000; fix with:"
    echo "  $DOCKER run --rm -v $VOLUME:/data --user root --entrypoint sh $IMAGE -c 'chown -R 1000:1000 /data'" >&2
    exit 1
fi

trap cleanup EXIT

say "1/2 one-shot selftest in $IMAGE"
"$DOCKER" run --rm $GPU_ARGS -v "$VOLUME:/data" $SMOKE_ENV --entrypoint python3 "$IMAGE" -m kindi selftest \
    || die "the one-shot selftest failed in $IMAGE (the FAIL line above names the step)"

say "2/2 the serving container, $NAME on 127.0.0.1:$PORT"
"$DOCKER" run -d --name "$NAME" $GPU_ARGS -p "127.0.0.1:$PORT:8093" -v "$VOLUME:/data" $SMOKE_ENV \
    -e KINDI_HOST=0.0.0.0 "$IMAGE" >/dev/null || die "could not start $NAME (docker rm -f $NAME if it exists)"
STARTED=1

waited=0
until "$DOCKER" exec "$NAME" python3 -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8093/health', timeout=3)" >/dev/null 2>&1; do
    if [ "$waited" -ge "$WAIT" ]; then
        "$DOCKER" logs --tail 40 "$NAME" || true
        die "no /health answer from $NAME in ${WAIT}s"
    fi
    sleep 2
    waited=$((waited + 2))
done
say "/health answers after ${waited}s"

"$DOCKER" exec "$NAME" python3 -m kindi selftest --url http://127.0.0.1:8093 \
    || die "the serving selftest failed in $NAME (the FAIL line above names the step)"

say "PASS: pack, tokenizer, engine, /health and the chat route in $IMAGE"
if [ "$KEEP" = 1 ]; then
    say "kept $NAME (127.0.0.1:$PORT); remove it with: docker rm -f $NAME"
fi
