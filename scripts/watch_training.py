from __future__ import annotations
import argparse,json,subprocess,time
from pathlib import Path


def _latest_record(path:Path):
    if not path.exists():return None
    last=None;last_steps=None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:event=json.loads(line)
        except json.JSONDecodeError:continue
        last=event
        if "steps_completed" in event:last_steps=event
    if last is not None and last_steps is not None and "steps_completed" not in last:return {**last_steps,**last}
    return last


def _gpu():
    try:
        row=subprocess.run(["nvidia-smi","--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=2).stdout.strip().splitlines()[0]
        name,util,used,total,power=(x.strip() for x in row.split(",",4))
        return f"{name} | GPU {util}% | VRAM {used}/{total} MiB | {power} W"
    except (OSError,subprocess.SubprocessError,IndexError,ValueError):return "GPU telemetry unavailable (nvidia-smi not found)"


def _render(run:Path,record,tty:bool):
    if tty:print("\x1b[2J\x1b[H",end="")
    print(f"OT Irregularity training monitor  |  {run}")
    if record is None:
        print("Status: waiting for training_progress.jsonl (data loading may still be running)")
    else:
        phase=record.get("phase","unknown");done=record.get("steps_completed");total=record.get("steps_requested")
        print(f"Phase: {phase}")
        if done is not None and total:
            ratio=max(0.,min(1.,done/total));width=36;bar="="*int(width*ratio)+">"+"."*max(0,width-int(width*ratio)-1)
            print(f"[{bar}] {done:,}/{total:,} updates ({ratio:.1%})")
        if "epochs_completed" in record:print(f"Epoch: {record['epochs_completed']:,}/{record.get('epochs_requested','?'):,} | best epoch: {record.get('best_epoch','?')}")
        if "best_validation_loss" in record:print(f"Best validation loss: {record['best_validation_loss']:.8g}")
        if "elapsed_seconds" in record or "training_seconds" in record:
            elapsed=record.get("elapsed_seconds",record.get("training_seconds"));eta=(elapsed*(total-done)/done) if done and total and done<total else 0
            print(f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        elif "training_seconds" in record:print(f"Training time: {record['training_seconds']:.1f}s")
    print(_gpu(),flush=True)


def main():
    parser=argparse.ArgumentParser(description="Live console monitor for OT Irregularity training")
    parser.add_argument("--run",required=True,type=Path,help="training output directory")
    parser.add_argument("--refresh",type=float,default=1.0,help="refresh period in seconds")
    args=parser.parse_args();progress=args.run/"training_progress.jsonl";tty=__import__("sys").stdout.isatty()
    while True:
        record=_latest_record(progress)
        if record is None:
            metadata=args.run/"training_metadata.json"
            if metadata.exists():
                data=json.loads(metadata.read_text(encoding="utf-8")).get("autoencoder_training",{});record={"phase":"completed",**data}
        _render(args.run,record,tty)
        if record and record.get("phase") in {"completed","autoencoder_completed"}:break
        time.sleep(max(.25,args.refresh))

if __name__=="__main__":main()
