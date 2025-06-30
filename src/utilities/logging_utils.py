import torch

def log_vram(status, logger):
    logger.info(
        str(status) + " - Current vram usage: %s GB",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
    )
