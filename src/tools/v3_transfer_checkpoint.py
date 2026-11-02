"""Atomic, recipe-bound checkpoints for our own local pilot training files."""
import os
from pathlib import Path


def atomic_save(path, payload):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_checkpoint(path, binding):
    import torch
    # Only load locally generated checkpoints, never downloaded pickle files.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('binding') != binding:
        raise ValueError('Checkpoint data/model/training recipe differs; refusing resume')
    return payload
