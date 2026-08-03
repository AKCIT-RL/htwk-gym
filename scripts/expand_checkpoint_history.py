"""Expand a soccer checkpoint's obs input dim by padding zeros (Frente F).

The task-obs history appends H*12 dims AFTER the existing 249, so padding the
input layers with zero columns (and obs-norm with mean 0 / std 1) keeps the
warm-start policy EXACTLY equivalent to the original on any state.

Usage (inside the mimickit container, from the MimicKit root):
    python3.8 expand_checkpoint_history.py --in_file A.pt --out_file B.pt \
        --old_dim 249 --new_dim 369
"""

import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_file", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--old_dim", type=int, default=249)
    p.add_argument("--new_dim", type=int, default=369)
    args = p.parse_args()

    sd = torch.load(args.in_file, map_location="cpu")
    pad = args.new_dim - args.old_dim
    assert pad > 0

    changed = []
    for k, v in sd.items():
        if (not torch.is_tensor(v)):
            continue
        if (v.dim() == 2 and v.shape[1] == args.old_dim):
            sd[k] = torch.cat([v, torch.zeros([v.shape[0], pad], dtype=v.dtype)], dim=1)
            changed.append((k, "weight cols +{:d} zeros".format(pad)))
        elif (v.dim() == 1 and v.shape[0] == args.old_dim):
            fill = 1.0 if ("._std" in k) else 0.0
            sd[k] = torch.cat([v, torch.full([pad], fill, dtype=v.dtype)])
            changed.append((k, "padded with {:.0f}".format(fill)))

    assert changed, "no tensor matched old_dim={:d}".format(args.old_dim)
    for k, what in changed:
        print("{:40s} {}".format(k, what))
    torch.save(sd, args.out_file)
    print("saved:", args.out_file)


if (__name__ == "__main__"):
    main()
