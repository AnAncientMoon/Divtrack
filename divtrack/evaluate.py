"""Evaluation with the Cell Tracking Challenge measures.

We use ``ctc_evaluate`` from the py-ctcmetrics package (version 1.3.3), which re-implements the
official CTC measures. The call is

    ctc_evaluate --gt <seq>_GT --res <result> --tra --det --lnk --ct --tf --bc 1 --cca --num-threads N

(``--lnk`` needs ``--tra``; without it py-ctcmetrics 1.3.3 returns an empty output.)
BIO uses BC(1), the division F1 score with a tolerance of one frame, so we report
OP_CLB = (LNK + BIO(1)) / 2 with BIO(1) = mean(CT, TF, BC(1), CCA).
"""
import os
import re
import shutil
import subprocess
import sys

KEYS = ("TRA", "DET", "LNK", "CT", "TF", "BC(1)", "CCA", "BIO(1)", "tp_div(1)", "fp_div(1)", "fn_div(1)")


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
    vals = {k: float(v) for k, v in re.findall(r"(\S+): ([-\d.eE]+)", "\n".join(lines))}
    if "LNK" not in vals or "BIO(1)" not in vals:
        raise RuntimeError("ctc_evaluate failed:\n" + "\n".join(r.stdout.splitlines()[-20:]))
    vals["OP_CLB"] = (vals["LNK"] + vals["BIO(1)"]) / 2.0
    return vals


def format_metrics(v):
    return ("OP_CLB %.5f | LNK %.5f BIO(1) %.5f | TRA %.5f DET %.5f | CT %.5f TF %.5f BC(1) %.5f CCA %.5f | divisions TP/FP/FN %d/%d/%d"
            % (v["OP_CLB"], v["LNK"], v["BIO(1)"], v["TRA"], v["DET"], v["CT"], v["TF"], v["BC(1)"], v["CCA"],
               v["tp_div(1)"], v["fp_div(1)"], v["fn_div(1)"]))
