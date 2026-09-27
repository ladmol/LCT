"""Train a reproducible vehicle identity model on the supplied archive.

Run with `python -m reid.train --archive data/dataset.zip` from ml/.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from eval.metrics import choose_refusal_threshold, refusal_metrics, retrieval_metrics
from reid.data import ContestDataset, IdentityBatchSampler, read_records
from reid.infer import embed_records
from reid.model import VehicleReID, batch_hard_triplet
from reid.protocol import cross_camera_protocol, split_identities


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--output", type=Path, default=Path("outputs/resnet50_256"))
    parser.add_argument(
        "--init-weights", type=Path, default=Path("data/weights/resnet50_imagenet.pth")
    )
    parser.add_argument(
        "--resume", type=Path, help="Continue from a compatible Re-ID checkpoint"
    )
    parser.add_argument(
        "--arch",
        choices=["resnet18", "resnet50", "convnext_tiny"],
        default="convnext_tiny",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--identities-per-batch", type=int, default=4)
    parser.add_argument("--views-per-identity", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--augmentation", choices=["basic", "strong"], default="basic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="For a quick smoke run; 0 means full epochs",
    )
    return parser.parse_args()


def evaluate(
    model, archive, records, identities, image_size, device, batch_size, workers, seed
):
    query, gallery = cross_camera_protocol(records, identities, seed)
    query_vectors = embed_records(
        model, archive, query, image_size, device, batch_size, workers
    )
    gallery_vectors = embed_records(
        model, archive, gallery, image_size, device, batch_size, workers
    )
    query_ids = [record.vehicle_id for record in query]
    gallery_ids = [record.vehicle_id for record in gallery]
    ranking = retrieval_metrics(
        query_vectors,
        gallery_vectors,
        query_ids,
        gallery_ids,
        [record.camera_id for record in query],
        [record.camera_id for record in gallery],
    )
    return ranking, query_vectors, gallery_vectors, query_ids, gallery_ids


def main():
    args = parse_args()
    if args.epochs < 1 or args.image_size < 64 or args.lr <= 0:
        raise ValueError("epochs, image-size, and learning rate must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    records = read_records(args.archive, "train.csv")
    splits = split_identities(records, args.seed)
    (args.output / "split.json").write_text(
        json.dumps(splits, indent=2), encoding="utf-8"
    )
    train_ids = set(splits["train"])
    train_records = [record for record in records if record.vehicle_id in train_ids]
    labels = {identity: index for index, identity in enumerate(sorted(train_ids))}
    sampler = IdentityBatchSampler(
        train_records, args.identities_per_batch, args.views_per_identity, args.seed
    )
    train_loader = DataLoader(
        ContestDataset(
            args.archive,
            train_records,
            args.image_size,
            training=True,
            augmentation=args.augmentation,
            label_to_index=labels,
        ),
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = VehicleReID(args.arch, len(labels))
    resumed = None
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=True)
        if (resumed["arch"], resumed["image_size"], resumed["num_classes"]) != (
            args.arch,
            args.image_size,
            len(labels),
        ):
            raise ValueError(
                "Resume checkpoint architecture, image size, or class count differs"
            )
        model.load_state_dict(resumed["model"])
        print(f"Resumed model: {args.resume}", flush=True)
    elif args.init_weights.is_file():
        state = torch.load(args.init_weights, map_location="cpu", weights_only=True)
        if "model" in state:
            state = {
                key.removeprefix("backbone."): value
                for key, value in state["model"].items()
                if key.startswith("backbone.")
            }
        incompatible = model.backbone.load_state_dict(state, strict=False)
        expected_head = (
            {"fc.weight", "fc.bias"}
            if args.arch.startswith("resnet")
            else {"classifier.2.weight", "classifier.2.bias"}
        )
        if incompatible.missing_keys or set(incompatible.unexpected_keys) not in (
            set(),
            expected_head,
        ):
            raise ValueError(f"Unexpected pretrained weights: {incompatible}")
        print(f"Loaded pretrained backbone: {args.init_weights}", flush=True)
    else:
        print(
            f"WARNING: {args.init_weights} missing; training the backbone from scratch",
            flush=True,
        )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_map = float(resumed["dev_map"]) if resumed else -1.0
    best_path = args.output / "best.pt"
    if resumed and args.resume.resolve() != best_path.resolve():
        torch.save(resumed, best_path)
    print(
        f"device={device} train_images={len(train_records)} train_ids={len(labels)} batches={len(train_loader)}",
        flush=True,
    )
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for step, (images, targets, _) in enumerate(train_loader):
            images, targets = (
                images.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                embedding, logits = model(images)
                if logits is None:
                    raise RuntimeError("Classifier missing during training")
                classification = F.cross_entropy(logits, targets)
                metric = batch_hard_triplet(embedding, targets)
                loss = classification + 0.5 * metric
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if args.max_steps and step + 1 >= args.max_steps:
                break
        model.eval()
        ranking, *_ = evaluate(
            model,
            args.archive,
            records,
            splits["dev"],
            args.image_size,
            device,
            args.eval_batch_size,
            args.workers,
            args.seed + 1,
        )
        current_epoch = (int(resumed["epoch"]) if resumed else 0) + epoch + 1
        print(
            f"epoch={current_epoch} loss={np.mean(losses):.4f} dev={ranking}",
            flush=True,
        )
        if ranking["mAP"] > best_map:
            best_map = ranking["mAP"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": args.arch,
                    "image_size": args.image_size,
                    "num_classes": len(labels),
                    "epoch": current_epoch,
                    "dev_map": best_map,
                    "seed": args.seed,
                    "augmentation": args.augmentation,
                },
                best_path,
            )
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    dev_rank, dev_query, dev_gallery, dev_qids, dev_gids = evaluate(
        model,
        args.archive,
        records,
        splits["dev"],
        args.image_size,
        device,
        args.eval_batch_size,
        args.workers,
        args.seed + 1,
    )
    threshold_result = choose_refusal_threshold(
        dev_query, dev_gallery, dev_qids, dev_gids
    )
    threshold = threshold_result["threshold"]
    holdout_rank, holdout_query, holdout_gallery, holdout_qids, holdout_gids = evaluate(
        model,
        args.archive,
        records,
        splits["holdout"],
        args.image_size,
        device,
        args.eval_batch_size,
        args.workers,
        args.seed + 2,
    )
    holdout_refusal = refusal_metrics(
        holdout_query, holdout_gallery, holdout_qids, holdout_gids, threshold
    )
    checkpoint["refusal_threshold"] = threshold
    torch.save(checkpoint, best_path)
    report = {
        "checkpoint": str(best_path),
        "dev_ranking": dev_rank,
        "dev_refusal": threshold_result,
        "holdout_ranking": holdout_rank,
        "holdout_refusal": holdout_refusal,
        "note": "Local protocol uses one gallery camera per known identity; hidden evaluation may differ.",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
