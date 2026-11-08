"""Pin probes: a full copy of the clean clone's research tree per case, one pinned file changed."""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path
CLONE = Path(sys.argv[1]); WORK = Path(sys.argv[2]); PY = sys.executable
S = "research/ti_tnn/scaling"
cases = [("pins/dbe_new_eval.json: one task id replaced", "pins/dbe_new_eval.json", lambda t: t.replace("bird23tf-01663", "bird23tf-00001", 1)),
         ("pins/dbe_protocol.json: protocol sha changed", "pins/dbe_protocol.json", lambda t: t.replace("89d877fb", "00000000", 1)),
         ("docs/PROTOCOL_database_evidence_revision.md: one character", "docs/PROTOCOL_database_evidence_revision.md", lambda t: t.replace("Protocol", "protocol", 1)),
         ("db_evidence.py: a comment added", "db_evidence.py", lambda t: t + "\n# changed\n"),
         ("dbe_run.py: a comment added", "dbe_run.py", lambda t: t + "\n# changed\n")]
out = []
for name, rel, fn in cases:
    d = Path(tempfile.mkdtemp(dir=WORK))
    shutil.copytree(CLONE / "research", d / "research")
    p = d / S / rel
    p.write_text(fn(p.read_text()))
    r = subprocess.run([PY, "-B", "-c", "import sys; sys.path.insert(0, sys.argv[1]);"
                        "from scaling import dbe_report as DR\n"
                        "try:\n DR.reconstruct(); print('accepted')\n"
                        "except DR.Inconsistent as e: print('refused:', str(e)[:150])", str(d / "research/ti_tnn")],
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""})
    res = (r.stdout.strip() or ("CRASH " + r.stderr.strip()[-200:]))
    out.append({"case": name, "outcome": res, "as_expected": res.startswith("refused")})
    print(("OK  " if res.startswith("refused") else "BAD ") + name + ": " + res)
    shutil.rmtree(d)
(WORK / "pin_probe_results.json").write_text(json.dumps(out, indent=1))
