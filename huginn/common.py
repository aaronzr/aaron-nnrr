"""Small shared helpers for the experiment scripts."""

import argparse
import json
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="only use the first N items (quick test)")
    p.add_argument("--tiny", action="store_true",
                   help="use a tiny randomly initialized Huginn for testing the code (no download, meaningless results)")
    return p


def get_runner(args):
    from huginn_harness import Runner, load_huginn
    if args.tiny:
        from tests.tiny_huginn import load_tiny
        model, tok = load_tiny()
        model.to(args.device)
    else:
        dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
        model, tok = load_huginn(device=args.device, dtype=dtype)
    return Runner(model, tok)


def load_json(name_or_path):
    path = name_or_path if os.path.exists(name_or_path) else os.path.join(DATA, name_or_path)
    with open(path) as f:
        return json.load(f)


def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def results_path(name):
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def progress(iterable, desc=""):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc)
    except ImportError:
        return iterable


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    return pyplot


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t
