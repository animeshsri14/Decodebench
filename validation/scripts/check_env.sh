#!/usr/bin/env bash
# check_env.sh — verify GPU environment for DecodeBench validation
# PASS/FAIL lines, nonzero exit on hard failure.
set -euo pipefail

if [ -d "/usr/lib/wsl/lib" ]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi

FAILURES=0
WARNINGS=0

pass()  { echo "PASS: $*"; }
fail()  { echo "FAIL: $*"; FAILURES=$((FAILURES + 1)); }
warn()  { echo "WARN: $*"; WARNINGS=$((WARNINGS + 1)); }

echo "=== DecodeBench Environment Check ==="
echo

# --- Check 1: nvcc >= 12.0 ---
echo "--- nvcc version ---"
if command -v nvcc &>/dev/null; then
  NVCC_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
  if [ -n "$NVCC_VER" ]; then
    MAJOR=$(echo "$NVCC_VER" | cut -d. -f1)
    if [ "$MAJOR" -ge 12 ]; then
      pass "nvcc $NVCC_VER >= 12.0"
    else
      fail "nvcc $NVCC_VER < 12.0 (need >= 12.0)"
    fi
  else
    fail "Could not parse nvcc version"
  fi
else
  fail "nvcc not found"
fi

# --- Check 2: GPU visible ---
echo "--- GPU detection ---"
if command -v nvidia-smi &>/dev/null; then
  GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$GPU_COUNT" -gt 0 ]; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    pass "GPU visible: $GPU_NAME (count=$GPU_COUNT)"
  else
    fail "nvidia-smi reports 0 GPUs"
  fi
else
  fail "nvidia-smi not found"
fi

# --- Check 3: ncu permission ---
echo "--- NCU permissions ---"
if command -v ncu &>/dev/null; then
  # Quick test: run ncu with minimal metrics on a trivial kernel
  # Use a short timeout of 5s and check exit code/output
  NCU_OUT=$(timeout 5s ncu --metrics dram__bytes_read.sum --launch-skip 0 --launch-count 1 \
    /bin/true 2>&1) || true
  if echo "$NCU_OUT" | grep -qi "insufficient permissions\|permission denied\|Cannot access"; then
    fail "ncu lacks permissions (try sudo or admin group)"
  else
    pass "ncu available"
  fi
else
  fail "ncu not found"
fi

# --- Check 4: clock-lock capability (warn only) ---
echo "--- Clock locking ---"
if command -v nvidia-smi &>/dev/null; then
  # Try reading current clock lock status
  if nvidia-smi -q -d CLOCK 2>/dev/null | grep -qi "locked"; then
    pass "GPU clock locking supported"
  else
    warn "GPU clock locking may not be available (not a hard failure)"
  fi
else
  warn "Cannot check clock locking (nvidia-smi missing)"
fi

# --- Check 5: free memory >= 6 GB ---
echo "--- GPU memory ---"
if command -v nvidia-smi &>/dev/null; then
  FREE_MEM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$FREE_MEM_MB" ]; then
    FREE_GB=$(echo "scale=1; $FREE_MEM_MB / 1024" | bc 2>/dev/null || echo "0")
    if [ "${FREE_MEM_MB:-0}" -ge 6144 ]; then
      pass "Free GPU memory: ${FREE_GB} GB >= 6 GB"
    else
      fail "Free GPU memory: ${FREE_GB} GB < 6 GB"
    fi
  else
    fail "Cannot query GPU memory"
  fi
else
  fail "nvidia-smi not found"
fi

echo
echo "=== Summary: $FAILURES failure(s), $WARNINGS warning(s) ==="

if [ "$FAILURES" -gt 0 ]; then
  exit 1
fi
exit 0
