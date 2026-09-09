#!/usr/bin/env python3
import argparse, json, subprocess
from pathlib import Path

MODES=["baseline","persist","prefetch","combined"]

def run_one(binary,mode,extra):
    p=subprocess.run([str(binary),mode,*extra],capture_output=True,text=True)
    if p.returncode:
        raise RuntimeError(f"{mode} failed:\n{p.stdout}\n{p.stderr}")
    return json.loads(p.stdout.strip().splitlines()[-1])

def close(a,b,rel=1e-8,abs_=1e-6):
    return abs(a-b)<=max(abs_,rel*max(abs(a),abs(b),1.0))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--binary",required=True,type=Path)
    ap.add_argument("--output",default="eval-result.json",type=Path)
    ap.add_argument("extra",nargs=argparse.REMAINDER)
    a=ap.parse_args(); extra=a.extra[1:] if a.extra and a.extra[0]=="--" else a.extra
    results={m:run_one(a.binary,m,extra) for m in MODES}
    base=results["baseline"]; checksum=base["checksum"]
    for m,r in results.items():
        if not close(checksum,r["checksum"]):
            raise SystemExit(f"correctness failure: {m} checksum {r['checksum']} != {checksum}")
    base_ms=base["elapsed_ms"]
    speedups={m:base_ms/r["elapsed_ms"] for m,r in results.items()}
    best=max((m for m in MODES if m!="baseline"),key=lambda m:speedups[m])
    out={"schema_version":1,"correctness":"pass","results":results,
         "speedup_vs_baseline":speedups,"best_mode":best,
         "best_synthetic_gain_pct":(speedups[best]-1.0)*100.0,
         "interpretation":"synthetic feasibility only; do not publish as a model-serving speedup"}
    a.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
