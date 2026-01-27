import argparse
import signal
import time

import torch
import torch.multiprocessing as mp


def _dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"bad dtype: {name}")


def worker(rank: int, gpus: int, args, stop: mp.Event):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dt = _dtype(args.dtype)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    a = torch.randn(args.m, args.k, device=device, dtype=dt)
    b = torch.randn(args.k, args.n, device=device, dtype=dt)

    for _ in range(args.warmup):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize(device)

    it = 0
    t0 = time.time()
    while not stop.is_set():
        _ = torch.matmul(a, b)
        it += 1
        if args.print_every > 0 and it % args.print_every == 0:
            torch.cuda.synchronize(device)
            dt_s = time.time() - t0
            print(f"[gpu {rank}/{gpus}] iters={it} last={dt_s:.3f}s", flush=True)
            t0 = time.time()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--print-every", type=int, default=200)
    p.add_argument("--tf32", action="store_true")
    args = p.parse_args()

    ndev = torch.cuda.device_count()
    if ndev <= 0:
        raise SystemExit("no cuda")
    gpus = min(args.gpus, ndev)

    mp.set_start_method("spawn", force=True)
    stop = mp.Event()

    def _sig(*_):
        stop.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    procs = []
    for r in range(gpus):
        proc = mp.Process(target=worker, args=(r, gpus, args, stop))
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
