"""Heartbeat monitor untuk antrian Group A. Cetak satu blok status ringkas:
- jumlah run selesai (metrics.json) dari 16
- run terakhir yang selesai + waktunya
- GPU: util & mem, apakah ada proses python di GPU (training hidup?)
- keep-awake hidup?

    python src/models/opensource/monitor_groupA.py
"""
from __future__ import annotations
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "groupA"
WIB = timezone(timedelta(hours=7))

ALL = [(d, a) for d in ("nusax", "finance", "ugm", "smsa")
       for a in ("indobert-base", "lightweight", "indobertweet", "indobert-large")]


def gpu_status() -> str:
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        util, used, total = [x.strip() for x in q.stdout.strip().split(",")]
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        procs = [l for l in p.stdout.strip().splitlines() if l.strip()]
        return f"util={util}% mem={used}/{total}MiB | proc_di_gpu={len(procs)}"
    except Exception as e:
        return f"(nvidia-smi gagal: {e})"


def main() -> None:
    done = [(d, a) for d, a in ALL if (RESULTS / d / a / "metrics.json").exists()]
    mtimes = [(RESULTS / d / a / "metrics.json").stat().st_mtime for d, a in done]
    last = max(mtimes) if mtimes else None
    now = datetime.now(WIB)
    print(f"[{now:%Y-%m-%d %H:%M WIB}] Group A: {len(done)}/16 selesai")
    if last:
        lt = datetime.fromtimestamp(last, WIB)
        age = (now - lt).total_seconds() / 60
        last_run = max(done, key=lambda da: (RESULTS / da[0] / da[1] / "metrics.json").stat().st_mtime)
        print(f"  run terakhir selesai: {last_run[1]} x {last_run[0]}  ({lt:%H:%M WIB}, {age:.0f} mnt lalu)")
    pending = [f"{a}x{d}" for d, a in ALL if (d, a) not in done]
    print(f"  belum: {', '.join(pending) if pending else '(tidak ada — SELESAI)'}")
    print(f"  GPU: {gpu_status()}")


if __name__ == "__main__":
    main()
