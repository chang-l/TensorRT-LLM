import multiprocessing as mp
import os

import torch


def work():
    print("child pid", os.getpid())
    torch.cuda.set_device(0)
    x = torch.randn(64, 64, device="cuda")
    z = x @ x
    print("child matmul:", z.sum().item())


if __name__ == "__main__":
    print("parent pid", os.getpid())
    torch.cuda.set_device(0)
    x = torch.randn(64, 64, device="cuda")
    print("parent matmul:", (x @ x).sum().item())
    mp.set_start_method("spawn", force=True)
    p = mp.Process(target=work)
    p.start()
    p.join()
    print("exit:", p.exitcode)
