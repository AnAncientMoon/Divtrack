"""Evaluation with the Cell Tracking Challenge measures.

We use ``ctc_evaluate`` from the py-ctcmetrics package (version 1.3.3), which re-implements the
official CTC measures. The call is

    ctc_evaluate --gt <seq>_GT --res <result> --tra --det --lnk --ct --tf --bc 1 --cca --num-threads N

(``--lnk`` needs ``--tra``; without it py-ctcmetrics 1.3.3 returns an empty output.)
BIO uses BC(1), the division F1 score with a tolerance of one frame, so we report
OP_CLB = (LNK + BIO(1)) / 2 with BIO(1) = mean(CT, TF, BC(1), CCA). Measures that ctc_evaluate prints as N/A
(BC and CCA of a sequence without divisions) are returned as NaN, printed as n/a and written as null in JSON files.
"""
import math
import os
import re
import shutil
import subprocess
import sys

KEYS = ("TRA", "DET", "LNK", "CT", "TF", "BC(1)", "CCA", "BIO(1)", "tp_div(1)", "fp_div(1)", "fn_div(1)")



def json_safe(obj):
    """Copy of ``obj`` with NaN and infinities replaced by None (null): they are not valid JSON."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def find_ctc_evaluate():
    exe = os.path.join(os.path.dirname(sys.executable), "ctc_evaluate")
    return exe if os.path.exists(exe) else shutil.which("ctc_evaluate")


def evaluate(gt_dir, res_dir, threads=4):
    """Run ctc_evaluate and return a dict of the measures above plus OP_CLB."""
    exe = find_ctc_evaluate()
    if not exe:
        raise RuntimeError("ctc_evaluate not found; install py-ctcmetrics==1.3.3")
    cmd = [exe, "--gt", os.path.abspath(gt_dir), "--res", os.path.abspath(res_dir), "--tra", "--det", "--lnk",
           "--ct", "--tf", "--bc", "1", "--cca", "--num-threads", str(threads)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = [l for l in r.stdout.splitlines() if re.match(r"^(%s):" % "|".join(re.escape(k) for k in KEYS), l)]
    vals = {k: float("nan") if v == "N/A" else float(v)
            for k, v in re.findall(r"(\S+): (N/A|[-\d.eE]+)", "\n".join(lines))}
    if "LNK" not in vals or "BIO(1)" not in vals:
        raise RuntimeError("ctc_evaluate failed:\n" + "\n".join(r.stdout.splitlines()[-20:]))
    vals["OP_CLB"] = (vals["LNK"] + vals["BIO(1)"]) / 2.0
    return vals


def format_metrics(v):
    def num(key, fmt="%.5f"):
        x = v.get(key, float("nan"))
        return "n/a" if x != x else fmt % x

    return ("OP_CLB %s | LNK %s BIO(1) %s | TRA %s DET %s | CT %s TF %s BC(1) %s CCA %s | divisions TP/FP/FN %s/%s/%s"
            % (num("OP_CLB"), num("LNK"), num("BIO(1)"), num("TRA"), num("DET"), num("CT"), num("TF"), num("BC(1)"),
               num("CCA"), num("tp_div(1)", "%d"), num("fp_div(1)", "%d"), num("fn_div(1)", "%d")))
