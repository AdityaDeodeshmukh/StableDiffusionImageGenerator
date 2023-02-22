import torch
def tune(opt,pipe):
    torch.backends.cudnn.benchmark = True
    for i in opt:
        if i=="v":
            pipe.enable_vae_slicing()
            continue
        if i=="o":
            pipe.enable_sequential_cpu_offload()
            continue
        if i=="a":
            pipe.enable_attention_slicing(1)
            continue
    return pipe