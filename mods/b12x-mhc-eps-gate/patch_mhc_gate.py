#!/usr/bin/env python3
"""Patch b12x/norm/mhc/_impl.py so the fused Gram mHC kernel accepts
rms_norm_eps=1e-20 (DeepSeek-V4-Flash-Vision-Exp ships 1e-20; the 0731 model
ships 1e-6). Without this, vision-exp decode falls to the per-layer tilelang
mHC path (43 layers x 3 tilelang ops per decode step).

Why safe: the fused Gram kernel numerics are verified offline for
rms_eps <= 1e-6 (run_mhc_pre_partial + run_mhc_finalize_gram with rms_eps=1e-20
produce finite output matching rms_eps=1e-6: y_norm=88.5 / comb=2.11).

Two b12x gate forms are handled (same file, upstream evolved over images):

  FORM A (newer, e.g. a50ebee1d 9/4 image): the gate is a membership test
      `and float(rms_eps) in MHC_SUPPORTED_RMS_EPS`
  against the module constant at the top of _impl.py:
      MHC_SUPPORTED_RMS_EPS = (1.0e-6, 1.0e-5)
  The two fused-gate sites (b12x_mhc_pre and b12x_mhc_post_pre) share that one
  constant, and the ValueError messages interpolate it, so widening the tuple
  once unblocks both sites and keeps the error text truthful.
      -> NEW: MHC_SUPPORTED_RMS_EPS = (1.0e-20, 1.0e-6, 1.0e-5)

  FORM B (older): two identical hard-coded sites
      `and float(rms_eps) == 1.0e-6`
      -> NEW: `and float(rms_eps) <= 1.0e-6`

Idempotent: patched lines carry the marker "# b12x-mhc-eps-gate";
--check verifies the active form is patched and no unpatched sites remain.
"""
import sys

MARKER = "# b12x-mhc-eps-gate"

# FORM A (constant tuple gate)
A_OLD = "MHC_SUPPORTED_RMS_EPS = (1.0e-6, 1.0e-5)"
A_NEW = (
    "MHC_SUPPORTED_RMS_EPS = (1.0e-20, 1.0e-6, 1.0e-5)  "
    + MARKER
    + ": accept rms_norm_eps<=1e-6 (vision-exp=1e-20)"
)

# FORM B (hard-coded == gate, older images)
B_OLD = "        and float(rms_eps) == 1.0e-6"
B_NEW = (
    "        and float(rms_eps) <= 1.0e-6  "
    + MARKER
    + ": accept rms_norm_eps<=1e-6 (vision-exp=1e-20)"
)


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    check = "--check" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--check"]
    if len(args) != 1:
        print("usage: patch_mhc_gate.py [--check] <_impl.py>", file=sys.stderr)
        return 2
    path = args[0]
    text = load(path)

    a_old_count = text.count(A_OLD)
    b_old_count = text.count(B_OLD)
    marker_count = text.count(MARKER)

    if check:
        # Patched = no unpatched site of the form actually present, and at
        # least one marker. Form A patches 1 constant; form B patches 2 sites.
        if a_old_count == 0 and b_old_count == 0 and marker_count >= 1:
            print(
                "check OK: gate patched (marker=%d), 0 unpatched sites"
                % marker_count
            )
            return 0
        print(
            "check FAIL: formA_sites=%d formB_sites=%d marker=%d"
            % (a_old_count, b_old_count, marker_count),
            file=sys.stderr,
        )
        return 1

    if a_old_count == 0 and b_old_count == 0 and marker_count == 0:
        print(
            "FAIL: neither gate form found "
            "(expected MHC_SUPPORTED_RMS_EPS tuple or '== 1.0e-6' site; "
            "did upstream b12x change?)",
            file=sys.stderr,
        )
        return 1

    if a_old_count == 0 and b_old_count == 0:
        print("already patched (marker=%d), no-op" % marker_count)
        return 0

    changed = 0
    if a_old_count:
        text = text.replace(A_OLD, A_NEW)
        changed += 1
    if b_old_count:
        text = text.replace(B_OLD, B_NEW)
        changed += 1
    save(path, text)
    print(
        "patched gate (form A constant: %d, form B sites: %d)" % (a_old_count, b_old_count)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
