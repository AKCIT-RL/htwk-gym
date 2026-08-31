"""Compare MimicKit training logs column-by-column, matched on Samples.

Two uses:

  Inertness smoke (E0). Prove a code change did not alter training: run the
  BASELINE config on the patched code and check the shared columns reproduce
  the reference run. New columns are reported as added, not as failures.

      python3 scripts/compare_train_logs.py \\
          --ref  MimicKit/output/olives_seed2/log.txt \\
          --new  MimicKit/output/e0_inercia/log.txt \\
          --mode inertness

  A/B verdict. Read the S2 gate off two finished runs.

      python3 scripts/compare_train_logs.py \\
          --ref  MimicKit/output/olives_seed2/log.txt \\
          --new  MimicKit/output/s2_e1_seed2/log.txt \\
          --mode ab

`log.txt` is a single whitespace-separated stream: the leading non-numeric
tokens are the header, everything after is row-major values.
"""

import argparse
import sys

# S2 verdict gates (plano-fases2-4.md section 03). sense: "up" = higher is
# better, "down" = lower is better, "flat" = report only, no verdict.
AB_METRICS = [
    ("Test_Return", "up"),
    ("Test_Episode_Length", "up"),
    ("Approx_Kl", "flat"),
    ("Actor_Lr_Final", "flat"),
    ("Clip_Frac", "flat"),
    ("Imp_Ratio", "flat"),
    ("Disc_Reward_Mean", "flat"),
    ("Disc_Reward_Std", "up"),
    ("Adv_Goal_Std", "flat"),
    ("Adv_Aux_Std", "flat"),
    ("Critic_Loss", "flat"),
    ("Actor_Loss", "flat"),
]


def _is_number(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def read_log(path):
    """-> (header list, list of row dicts)."""
    with open(path) as f:
        tokens = f.read().split()

    header = []
    for tok in tokens:
        if _is_number(tok):
            break
        header.append(tok)

    if not header:
        raise SystemExit("no header found in {}".format(path))

    values = tokens[len(header):]
    ncol = len(header)
    if len(values) % ncol != 0:
        # a run killed mid-write leaves a partial row; drop it rather than
        # silently shifting every column
        values = values[:len(values) - (len(values) % ncol)]
        print("warning: {} ends with a partial row, dropped".format(path))

    rows = []
    for i in range(0, len(values), ncol):
        rows.append(dict(zip(header, [float(v) for v in values[i:i + ncol]])))
    return header, rows


def by_samples(rows):
    return {int(r["Samples"]): r for r in rows}


def fmt(v):
    if v is None:
        return "-"
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e6):
        return "{:.3e}".format(v)
    return "{:.6g}".format(v)


def mode_inertness(ref_header, ref_rows, new_header, new_rows, rel_tol):
    ref_by = by_samples(ref_rows)
    new_by = by_samples(new_rows)

    shared_samples = sorted(set(ref_by) & set(new_by))
    if not shared_samples:
        raise SystemExit("no Samples value appears in both logs - nothing to compare")

    added = [c for c in new_header if c not in ref_header]
    removed = [c for c in ref_header if c not in new_header]
    compared = [c for c in ref_header if c in new_header and c != "Samples"]

    # Wall_Time and Samples_Per_Second measure the machine, not the training
    volatile = {"Wall_Time", "Samples_Per_Second"}
    compared = [c for c in compared if c not in volatile]

    print("Rows matched on Samples: {}".format(
        ", ".join(str(s) for s in shared_samples)))
    if added:
        print("Columns added by the change (not a failure): {}".format(", ".join(added)))
    if removed:
        print("Columns REMOVED by the change: {}".format(", ".join(removed)))
    print("Ignored as machine-dependent: {}".format(", ".join(sorted(volatile))))
    print()

    failures = []
    for samples in shared_samples:
        r, n = ref_by[samples], new_by[samples]
        bad = []
        for col in compared:
            rv, nv = r[col], n[col]
            scale = max(abs(rv), abs(nv))
            if scale == 0.0:
                ok = True
            else:
                ok = abs(rv - nv) / scale <= rel_tol
            if not ok:
                bad.append((col, rv, nv))

        status = "PASS" if not bad else "FAIL"
        print("[{}] Samples={}  ({}/{} columns within {:.0%})".format(
            status, samples, len(compared) - len(bad), len(compared), rel_tol))
        for col, rv, nv in bad:
            print("       {:<24} ref={:<14} new={:<14} rel={:.3%}".format(
                col, fmt(rv), fmt(nv), abs(rv - nv) / max(abs(rv), abs(nv))))
        failures.extend(bad)

    print()
    if failures:
        print("INERTNESS FAILED: the change altered training. Do not run any A/B "
              "until this is understood.")
        return 1
    print("INERTNESS OK: shared columns reproduce the reference. The default "
          "code path is inert.")
    return 0


def mode_ab(ref_rows, new_rows, ref_name, new_name):
    r, n = ref_rows[-1], new_rows[-1]
    print("Final row: ref Samples={}  |  new Samples={}".format(
        int(r["Samples"]), int(n["Samples"])))
    if int(r["Samples"]) != int(n["Samples"]):
        print("warning: runs ended at different sample counts - not a strict A/B")
    print()

    head = "{:<22} {:>14} {:>14} {:>12}   {}".format(
        "Metric", ref_name[:14], new_name[:14], "delta %", "verdict")
    print(head)
    print("-" * len(head))

    regressions = 0
    for col, sense in AB_METRICS:
        rv = r.get(col)
        nv = n.get(col)
        if rv is None and nv is None:
            continue
        if rv is None or nv is None:
            print("{:<22} {:>14} {:>14} {:>12}   {}".format(
                col, fmt(rv), fmt(nv), "-", "only in one run"))
            continue

        delta = "-" if rv == 0 else "{:+.2f}".format(100.0 * (nv - rv) / abs(rv))
        if sense == "up":
            verdict = "ok" if nv >= rv else "REGRESSION"
        elif sense == "down":
            verdict = "ok" if nv <= rv else "REGRESSION"
        else:
            verdict = ""
        if verdict == "REGRESSION":
            regressions += 1
        print("{:<22} {:>14} {:>14} {:>12}   {}".format(
            col, fmt(rv), fmt(nv), delta, verdict))

    print()
    # gates that are absolute, not relative to the reference
    drs = n.get("Disc_Reward_Std")
    if drs is not None:
        print("Disc_Reward_Std > 0.1 (style collapse gate): {} ({})".format(
            "PASS" if drs > 0.1 else "FAIL", fmt(drs)))

    lr = n.get("Actor_Lr_Final")
    if lr is not None:
        pinned = ""
        if abs(lr - 1e-5) / 1e-5 < 1e-6:
            pinned = "  <- pinned at lr_min, controller mis-calibrated"
        elif abs(lr - 1e-2) / 1e-2 < 1e-6:
            pinned = "  <- pinned at lr_max, controller mis-calibrated"
        print("Actor_Lr_Final: {}{}".format(fmt(lr), pinned))

    kl = n.get("Approx_Kl")
    if kl is not None:
        print("Approx_Kl: {} (target 0.01, dead band 0.005-0.02)".format(fmt(kl)))

    print()
    print("Viewer verdict (2 lines) is still required by the briefing protocol.")
    return 1 if regressions else 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="reference log.txt (the baseline)")
    p.add_argument("--new", required=True, help="log.txt under test")
    p.add_argument("--mode", choices=["inertness", "ab"], default="inertness")
    p.add_argument("--rel-tol", type=float, default=1e-4,
                   help="relative tolerance for inertness mode (default 1e-4)")
    args = p.parse_args()

    ref_header, ref_rows = read_log(args.ref)
    new_header, new_rows = read_log(args.new)
    if not ref_rows or not new_rows:
        raise SystemExit("one of the logs has no data rows")

    print("ref: {} ({} rows, {} columns)".format(args.ref, len(ref_rows), len(ref_header)))
    print("new: {} ({} rows, {} columns)".format(args.new, len(new_rows), len(new_header)))
    print()

    if args.mode == "inertness":
        return mode_inertness(ref_header, ref_rows, new_header, new_rows, args.rel_tol)
    return mode_ab(ref_rows, new_rows, "baseline", "new")


if __name__ == "__main__":
    sys.exit(main())
