#!/usr/bin/env python
"""S71 GPU scheduler: FIFO job queue onto cards showing < 2 GiB in nvidia-smi.

Queue file: one job per line, "name | command". Commands run with
CUDA_VISIBLE_DEVICES=<card> pinned by the scheduler; each command may chain
train && eval. Polls every 60 s. Never touches busy cards; skips cards used by
processes outside this scheduler (memory check covers that).

Usage: python s71_scheduler.py --queue s71_queue.txt [--dry]
State file: s71_scheduler_state.json (name -> {card, pid, status, rc}).
"""
import argparse
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "s71_scheduler_state.json")
PY = "<local>"
MEM_LIMIT_MIB = 2000
POLL_S = 60


def free_cards():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
        text=True)
    cards = {}
    for line in out.strip().splitlines():
        idx, mem = line.split(",")
        cards[int(idx)] = int(mem.strip().split()[0])
    return {i: m for i, m in cards.items() if m < MEM_LIMIT_MIB}


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    jobs = []
    for line in open(os.path.join(HERE, args.queue)):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, cmd = line.split("|", 1)
        jobs.append({"name": name.strip(), "cmd": cmd.strip()})

    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    print(f"[sched] {len(jobs)} jobs queued; state has {len(st)} entries",
          flush=True)
    pending = [j for j in jobs
               if st.get(j["name"], {}).get("status") not in ("done",
                                                              "finished?")]
    while pending:
        # reap finished (rc unreadable cross-process: the log tail is the source
        # of truth, checked by the operator; "finished?" jobs are never re-queued
        # automatically -- mark them "done" in the state file after verifying)
        for name, e in list(st.items()):
            if e.get("status") == "running" and not pid_alive(e["pid"]):
                e["status"] = "finished?"
                e["ended"] = time.time()
        json.dump(st, open(STATE, "w"), indent=1)
        running_cards = {e["card"] for e in st.values()
                         if e.get("status") == "running"}
        try:
            free = free_cards()
        except Exception as ex:
            print(f"[sched] nvidia-smi failed: {ex}; retry in {POLL_S}s",
                  flush=True)
            time.sleep(POLL_S)
            continue
        avail = sorted(c for c in free if c not in running_cards)
        if avail and pending:
            job = pending.pop(0)
            card = avail[0]
            logf = os.path.join(HERE, f"s71_{job['name']}.log")
            jcmd = job["cmd"].replace("@PY@", PY)
            cmd = (f"cd {HERE} && ( export CUDA_VISIBLE_DEVICES={card}; "
                   f"{jcmd} ) > {logf} 2>&1")
            print(f"[sched] launch {job['name']} on card {card}: {cmd}",
                  flush=True)
            if not args.dry:
                p = subprocess.Popen(["bash", "-c", cmd])
                st[job["name"]] = {"card": card, "pid": p.pid,
                                   "status": "running", "cmd": cmd,
                                   "started": time.time(), "log": logf}
                json.dump(st, open(STATE, "w"), indent=1)
            else:
                st[job["name"]] = {"card": card, "status": "dry"}
            continue
        time.sleep(POLL_S)
    json.dump(st, open(STATE, "w"), indent=1)
    print("[sched] queue drained", flush=True)


if __name__ == "__main__":
    main()
