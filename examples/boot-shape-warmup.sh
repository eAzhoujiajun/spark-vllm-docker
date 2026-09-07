#!/usr/bin/env bash
# Warm DeepSeek V4 DSpark/vLLM request shapes after the API becomes reachable.
# Non-fatal by default: failures mean some shapes may JIT during live traffic.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

_url_host="${VLLM_HOST:-${HOST:-127.0.0.1}}"
case "$_url_host" in
  0.0.0.0|::|"[::]"|"") _url_host=127.0.0.1 ;;
esac
if [[ "$_url_host" == *:* && "$_url_host" != \[*\] ]]; then
  _url_host="[$_url_host]"
fi
BASE="${1:-${WARMUP_BASE_URL:-http://${_url_host}:${VLLM_PORT:-${PORT:-8888}}}}"
MODEL="${2:-${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp}}"
CURL_BIN="${WARMUP_CURL:-curl}"
REQ_TIMEOUT="${DSPARK_WARMUP_REQ_TIMEOUT:-240}"
WAIT_SECONDS="${DSPARK_WARMUP_WAIT_SECONDS:-1800}"
MAX_CONCURRENCY="${DSPARK_WARMUP_MAX_CONCURRENCY:-${MAX_NUM_SEQS:-6}}"
# Sampler-cache postcondition cache root. The container's default Triton cache
# is ~/.triton, mounted back to the host's HOME (launch-cluster bind), so the
# host-side default $HOME/.triton reaches the served rank's cache on the head
# node. Unset or non-existent => the postcondition is SKIPPED (repo2 opts out
# rather than mis-verify).
DSPARK_WARMUP_TRITON_CACHE_DIR="${DSPARK_WARMUP_TRITON_CACHE_DIR:-$HOME/.triton}"

case "$MAX_CONCURRENCY" in
  ''|*[!0-9]*|0)
    echo "boot-shape-warmup: invalid max concurrency ${MAX_CONCURRENCY@Q}; using 6" >&2
    MAX_CONCURRENCY=6
    ;;
esac

AUTH_ARGS=()
if [ -n "${DSPARK_WARMUP_BEARER:-}" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${DSPARK_WARMUP_BEARER}")
elif [ -n "${VLLM_API_KEY:-}" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${VLLM_API_KEY}")
elif [ -n "${DSPARK_API_KEYS:-}" ]; then
  read -r -a _dspark_keys <<< "${DSPARK_API_KEYS}"
  if [ "${#_dspark_keys[@]}" -gt 0 ]; then
    AUTH_ARGS=(-H "Authorization: Bearer ${_dspark_keys[0]}")
  fi
fi

# Sampler-cache postcondition (MiaAI-Lab). The sampling arms above dispatch
# _topk_topp_kernel only when a request carries top_k and/or top_p, so without
# this check a cold boot can leave a combo JIT to mid-serve. Inspect the head
# rank's persistent Triton cache for the combo's constexpr specialisations.
SAMPLER_KERNEL=_topk_topp_kernel

sampler_cache_combos() { # $1 = triton cache root; emits one line per constexpr combo
  local root=$1
  for ttir in "$root"/*/"$SAMPLER_KERNEL.ttir"; do
    [ -f "$ttir" ] || continue
    if grep -q "TOPK_ENABLED[^0-9]*1" "$ttir" && grep -q "TOPP_ENABLED[^0-9]*1" "$ttir"; then
      echo k+p
    elif grep -q "TOPK_ENABLED[^0-9]*1" "$ttir"; then
      echo k-only
    elif grep -q "TOPP_ENABLED[^0-9]*1" "$ttir"; then
      echo p-only
    else
      echo neither
    fi
  done | sort -u
}

verify_sampler_cache() { # postcondition; returns 0 = met or skipped, 1 = unmet
  local root="${DSPARK_WARMUP_TRITON_CACHE_DIR:-}" combos combo n missing=""
  if [ -z "$root" ] || [ ! -d "$root" ]; then
    echo "  sampler-cache postcondition: SKIPPED (DSPARK_WARMUP_TRITON_CACHE_DIR unset or not a directory)"
    return 0
  fi
  combos=$(sampler_cache_combos "$root")
  for combo in k-only p-only k+p; do
    n=$(printf '%s\n' "$combos" | grep -cx "$combo")
    [ "$n" -ge 1 ] || missing="${missing} ${combo}:0/1"
  done
  if [ -z "$missing" ]; then
    echo "  sampler-cache postcondition: MET — ${SAMPLER_KERNEL} constexpr combos on this rank:"
    printf '%s\n' "$combos" | sed 's/^/    /'
    return 0
  fi
  echo "  sampler-cache postcondition: unmet —${missing} (constexpr combos)"
  return 1
}

wait_api() {
  local deadline=$((SECONDS + WAIT_SECONDS))
  while true; do
    if "$CURL_BIN" -fsS --max-time 10 "${AUTH_ARGS[@]}" "$BASE/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if (( SECONDS >= deadline )); then
      echo "boot-shape-warmup: API not reachable at $BASE after ${WAIT_SECONDS}s" >&2
      return 1
    fi
    sleep 5
  done
}

next_pow2() {
  local n=$1 p=1
  while [ "$p" -lt "$n" ]; do p=$((p * 2)); done
  printf '%s' "$p"
}

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

mk_prompt() {
  local words=$1 tag=$2 body
  body=$(printf 'warm %.0s' $(seq 1 "$words"))
  printf '[warmup %s] Ignore this filler context: %s Reply with OK.' "$tag" "$body"
}

post_json() {
  local endpoint=$1 payload=$2 out=$3
  if "$CURL_BIN" -fsS --max-time "$REQ_TIMEOUT" "${AUTH_ARGS[@]}" \
      "$BASE$endpoint" -H "Content-Type: application/json" \
      -d "$payload" >/dev/null 2>>"$tmpdir/errors"; then
    echo ok > "$out"
  else
    echo fail > "$out"
  fi
}

fire_chat() {
  local tag=$1 words=$2 thinking=$3 out=$4 profile=${5:-bounded} prompt payload sample_fields prompt_json
  prompt=$(mk_prompt "$words" "$tag")
  prompt_json=$(printf '%s' "$prompt" | json_escape)
  if [ "$profile" = "serve-default" ]; then
    payload='{"model":"'"$MODEL"'","messages":[{"role":"user","content":'"$prompt_json"'}],"temperature":0}'
  elif [ "${profile#sampling}" != "$profile" ]; then
    case "$profile" in
      sampling-k) sample_fields='"top_k":40' ;;
      sampling-p) sample_fields='"top_p":0.9' ;;
      *) sample_fields='"top_k":40,"top_p":0.9' ;;
    esac
    payload='{"model":"'"$MODEL"'","messages":[{"role":"user","content":'"$prompt_json"'}],"max_tokens":24,"temperature":0.8,'"$sample_fields"',"chat_template_kwargs":{"thinking":'"$thinking"',"reasoning_effort":"low"}}'
  else
    payload='{"model":"'"$MODEL"'","messages":[{"role":"user","content":'"$prompt_json"'}],"max_tokens":24,"temperature":0,"chat_template_kwargs":{"thinking":'"$thinking"',"reasoning_effort":"low"}}'
  fi
  post_json "/v1/chat/completions" "$payload" "$out"
}

burst() {
  local arm=$1 c=$2 words=$3 profile=${4:-bounded} i t0 t1
  for i in $(seq 1 "$c"); do : > "$tmpdir/${arm}-${i}"; done
  t0=$(date +%s)
  for i in $(seq 1 "$c"); do
    fire_chat "${arm}-${i}" "$words" true "$tmpdir/${arm}-${i}" "$profile" &
  done
  wait
  t1=$(date +%s)
  echo "  arm ${arm}: C=${c} x ~${words} tok, profile=${profile}, $((t1 - t0))s"
}

mk_ladder_prompt() {
  local n=$1 out="hello" i
  for ((i = 1; i < n; i++)); do out="$out hello"; done
  printf '%s' "$out"
}

verify_ladder_rung() {
  local s=$1 prompt prompt_json want_block got resp payload t0 t1
  : > "$tmpdir/ladder-$s"
  prompt=$(mk_ladder_prompt "$s")
  prompt_json=$(printf '%s' "$prompt" | json_escape)
  want_block=$(next_pow2 $((s + ${MTP_NUM_TOKENS:-6})))
  payload='{"model":"'"$MODEL"'","prompt":'"$prompt_json"'}'
  if ! resp=$("$CURL_BIN" -fsS --max-time 30 "${AUTH_ARGS[@]}" \
        "$BASE/tokenize" -H "Content-Type: application/json" \
        -d "$payload" 2>>"$tmpdir/errors"); then
    echo "  ladder s=${s}: tokenize failed, BLOCK ${want_block} not warmed"
    echo fail > "$tmpdir/ladder-$s"
    return 0
  fi
  got=$(printf '%s\n' "$resp" | grep -o '"count"[[:space:]]*:[[:space:]]*[0-9]*' | head -n 1 | grep -o '[0-9]*$')
  if [ -z "$got" ] || [ "$got" -ne "$s" ]; then
    echo "  ladder s=${s}: tokenize count ${got:-missing}/${s}, BLOCK ${want_block} skipped"
    echo fail > "$tmpdir/ladder-$s"
    return 0
  fi
  payload='{"model":"'"$MODEL"'","prompt":'"$prompt_json"',"max_tokens":1,"temperature":0}'
  t0=$(date +%s)
  post_json "/v1/completions" "$payload" "$tmpdir/ladder-$s"
  t1=$(date +%s)
  echo "  ladder s=${s}: tokenize ${got}/${s} -> BLOCK ${want_block} fired ($((t1 - t0))s)"
}

warm_image_once() {
  local out="$tmpdir/vision-image-1" image_url payload
  : > "$out"
  image_url="${DSPARK_WARMUP_IMAGE_URL:-https://raw.githubusercontent.com/vllm-project/vllm/main/examples/image1.jpeg}"
  payload='{"model":"'"$MODEL"'","messages":[{"role":"user","content":[{"type":"text","text":"Describe this image in one short sentence."},{"type":"image_url","image_url":{"url":"'"$image_url"'"}}]}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"thinking":false}}'
  post_json "/v1/chat/completions" "$payload" "$out"
  echo "  arm vision-image: image_url request fired"
}

if ! wait_api; then
  exit 1
fi

echo "boot-shape-warmup: sweeping DeepSeek V4 DSpark shapes at $BASE model=$MODEL"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
started=$(date +%s)

for s in 1 6 20 45 100 200; do
  verify_ladder_rung "$s"
done

EXPECTED_CHAT_REQUESTS=8
burst c1 1 300
burst short-c1 1 8 serve-default
burst samp-k 1 8 sampling-k
burst samp-p 1 8 sampling-p
burst samp-kp 1 8 sampling-kp
if [ "$MAX_CONCURRENCY" -ge 2 ]; then burst c2 2 420; burst short-c2 2 8 serve-default; EXPECTED_CHAT_REQUESTS=$((EXPECTED_CHAT_REQUESTS + 4)); fi
if [ "$MAX_CONCURRENCY" -ge 3 ]; then burst samp-k-c3 3 8 sampling-k; burst samp-p-c3 3 8 sampling-p; burst samp-kp-c3 3 8 sampling-kp; EXPECTED_CHAT_REQUESTS=$((EXPECTED_CHAT_REQUESTS + 9)); fi
if [ "$MAX_CONCURRENCY" -ge 4 ]; then burst c4 4 380; burst short-c4 4 8 serve-default; EXPECTED_CHAT_REQUESTS=$((EXPECTED_CHAT_REQUESTS + 8)); fi
if [ "$MAX_CONCURRENCY" -ge 6 ]; then burst c6 6 340; burst short-c6 6 8 serve-default; EXPECTED_CHAT_REQUESTS=$((EXPECTED_CHAT_REQUESTS + 12)); fi
burst mid 1 2600
burst longchunk 1 9500
: > "$tmpdir/nothink-1"
fire_chat nothink-1 300 false "$tmpdir/nothink-1"
echo "  arm nothink: C=1 x ~300 tok, thinking=false"
if [ "${DSPARK_WARMUP_VISION_IMAGE:-1}" != "0" ]; then
  warm_image_once
  EXPECTED_CHAT_REQUESTS=$((EXPECTED_CHAT_REQUESTS + 1))
fi

total=0
ok_count=0
for f in "$tmpdir"/*-*; do
  [ -f "$f" ] || continue
  total=$((total + 1))
  [ "$(cat "$f")" = ok ] && ok_count=$((ok_count + 1))
done
expected=$((6 + EXPECTED_CHAT_REQUESTS))
ended=$(date +%s)
echo "boot-shape-warmup: ${ok_count}/${total} requests ok in $((ended - started))s"
if [ "$total" -ne "$expected" ]; then
  echo "boot-shape-warmup: WARN tallied $total outcomes for $expected scheduled requests" >&2
fi
if [ "$ok_count" -lt "$total" ]; then
  echo "boot-shape-warmup: WARN $((total - ok_count)) request(s) failed; first errors:" >&2
  sed -n '1,8p' "$tmpdir/errors" >&2 2>/dev/null || true
  exit 1
fi

# Non-fatal sampler-cache postcondition: WARN on a miss, never fail the boot.
verify_sampler_cache || \
  echo "boot-shape-warmup: WARN sampler-cache postcondition unmet — a sampling arm may JIT cold later" >&2
exit 0
