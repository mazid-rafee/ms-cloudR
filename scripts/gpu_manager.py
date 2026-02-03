import argparse
import torch
import time
import os
import subprocess
import signal
import sys

def allocate_memory(bytes_to_allocate, gpu_id):
    torch.cuda.set_device(gpu_id)
    print(f"[ALLOC] Using GPU {gpu_id}")
    print(f"[ALLOC] Allocating {bytes_to_allocate / 1e9:.2f} GB on GPU...")
    tensor = torch.empty(int(bytes_to_allocate // 4), dtype=torch.float32, device=f'cuda:{gpu_id}')
    print(f"[ALLOC] Memory allocated and held on GPU {gpu_id}.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("[ALLOC] Received termination signal. Releasing memory.")
        del tensor
        torch.cuda.empty_cache()

def start_allocator_process(alloc_size_gb, gpu_id):
    pid_file = os.path.join(os.path.dirname(__file__), f"gpu_alloc_gpu{gpu_id}.pid")
    cmd = [sys.executable, __file__, "--_alloc_background", str(alloc_size_gb), "--gpu", str(gpu_id)]
    proc = subprocess.Popen(cmd)
    with open(pid_file, "w") as f:
        f.write(str(proc.pid))
    print(f"[ALLOC] Started background memory holder on GPU {gpu_id} with PID {proc.pid}")

def stop_allocator_process(gpu_id):
    pid_file = os.path.join(os.path.dirname(__file__), f"gpu_alloc_gpu{gpu_id}.pid")
    if not os.path.exists(pid_file):
        print(f"[FREE] No memory holder process found for GPU {gpu_id}.")
        return
    with open(pid_file, "r") as f:
        pid = int(f.read())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"[FREE] Terminated memory holder process (PID {pid}) for GPU {gpu_id}")
    except ProcessLookupError:
        print(f"[FREE] Process {pid} not running.")
    os.remove(pid_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alloc", nargs="?", const=15, type=float,
                        help="Allocate and hold GPU memory. Optionally specify GB (default: 15)")
    parser.add_argument("--free", action="store_true", help="Free previously allocated memory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to target (default: 0)")
    parser.add_argument("--_alloc_background", type=float, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.alloc is not None:
        start_allocator_process(alloc_size_gb=args.alloc, gpu_id=args.gpu)
    elif args.free:
        stop_allocator_process(gpu_id=args.gpu)
    elif args._alloc_background is not None:
        alloc_bytes = args._alloc_background * (1024 ** 3)
        allocate_memory(alloc_bytes, gpu_id=args.gpu)
    else:
        print("Usage:")
        print("  python scripts/gpu_manager.py --alloc [GB] --gpu [ID]   # Allocate and hold GPU memory (default 15GB on GPU 0)")
        print("  python scripts/gpu_manager.py --free --gpu [ID]         # Free the allocated memory on specified GPU")

if __name__ == "__main__":
    main()
