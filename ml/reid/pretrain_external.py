"""Pretrain instance embeddings and attribute heads on reviewed Auto.ru crops.

This stage never opens raw photos. Its checkpoint can initialize reid.train,
which then adapts the backbone to the contest's camera images.
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

from attributes.attribute_heads import AttributeHeads
from reid.data import read_records
from reid.external import ApprovedAutoDataset, ModelAwareBatchSampler, read_approved
from reid.model import VehicleReID, batch_hard_triplet
from reid.protocol import split_identities
from reid.train import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/auto_ru"))
    parser.add_argument(
        "--contest-archive", type=Path, default=Path("data/dataset.zip")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/external"))
    parser.add_argument(
        "--init-weights", type=Path, default=Path("data/weights/resnet18_imagenet.pth")
    )
    parser.add_argument(
        "--arch", choices=["resnet18", "resnet50", "convnext_tiny"], default="resnet18"
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--eval-every", type=int, default=1, help="Evaluate contest dev every N epochs"
    )
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--identities-per-batch", type=int, default=8)
    parser.add_argument("--views-per-identity", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.eval_every < 1:
        raise ValueError("epochs and eval-every must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = read_approved(args.data)
    if not records:
        raise ValueError(
            "No approved Auto.ru photos; review raw images before pretraining"
        )
    labels = {
        field: {
            value: index
            for index, value in enumerate(
                sorted(
                    {
                        getattr(record, field)
                        for record in records
                        if getattr(record, field) != "unknown"
                    }
                )
            )
        }
        for field in ("make", "model", "restyling")
    }
    labels["instance"] = {
        value: index
        for index, value in enumerate(
            sorted({record.instance_id for record in records})
        )
    }
    sampler = ModelAwareBatchSampler(
        records, args.identities_per_batch, args.views_per_identity, args.seed
    )
    loader = DataLoader(
        ApprovedAutoDataset(records, args.image_size, labels),
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = VehicleReID(args.arch, len(labels["instance"]))
    state = torch.load(args.init_weights, map_location="cpu", weights_only=True)
    incompatible = model.backbone.load_state_dict(state, strict=False)
    expected_head = (
        {"fc.weight", "fc.bias"}
        if args.arch.startswith("resnet")
        else {"classifier.2.weight", "classifier.2.bias"}
    )
    if incompatible.missing_keys or set(incompatible.unexpected_keys) != expected_head:
        raise ValueError(f"Unexpected pretrained weights: {incompatible}")
    heads = AttributeHeads(
        model.embedding_dim,
        max(1, len(labels["make"])),
        max(1, len(labels["model"])),
        max(1, len(labels["restyling"])),
    )
    model.to(device)
    heads.to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(heads.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    contest_records = (
        read_records(args.contest_archive, "train.csv")
        if args.contest_archive.is_file()
        else None
    )
    contest_dev = (
        split_identities(contest_records, args.seed)["dev"] if contest_records else None
    )
    args.output.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    for epoch in range(args.epochs):
        model.train()
        heads.train()
        losses = []
        for images, identity, make, model_label, restyling in loader:
            images = images.to(device, non_blocking=True)
            identity, make, model_label, restyling = (
                item.to(device, non_blocking=True)
                for item in (identity, make, model_label, restyling)
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                vectors, identity_logits = model(images)
                if identity_logits is None:
                    raise RuntimeError("Identity classifier missing")
                attributes = heads(vectors)
                loss = F.cross_entropy(
                    identity_logits, identity
                ) + 0.5 * batch_hard_triplet(vectors, identity)
                if (make >= 0).any():
                    loss += 0.2 * F.cross_entropy(
                        attributes["make"], make, ignore_index=-1
                    )
                if (model_label >= 0).any():
                    loss += 0.2 * F.cross_entropy(
                        attributes["model"], model_label, ignore_index=-1
                    )
                if len(labels["restyling"]) > 1 and (restyling >= 0).any():
                    loss += 0.1 * F.cross_entropy(
                        attributes["restyling"], restyling, ignore_index=-1
                    )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        score = -float(np.mean(losses))
        if (
            contest_records is not None
            and contest_dev is not None
            and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs)
        ):
            ranking, *_ = evaluate(
                model,
                args.contest_archive,
                contest_records,
                contest_dev,
                args.image_size,
                device,
                32,
                args.workers,
                args.seed + 1,
            )
            score = ranking["mAP"]
            print(
                f"external_epoch={epoch + 1} loss={np.mean(losses):.4f} contest_dev_mAP={score:.4f}",
                flush=True,
            )
        elif contest_records is not None:
            print(f"external_epoch={epoch + 1} loss={np.mean(losses):.4f}", flush=True)
            continue
        else:
            print(f"external_epoch={epoch + 1} loss={np.mean(losses):.4f}", flush=True)
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "attribute_heads": heads.state_dict(),
                    "attribute_labels": labels,
                    "arch": args.arch,
                    "image_size": args.image_size,
                    "num_classes": len(labels["instance"]),
                    "epoch": epoch + 1,
                    "selection_score": score,
                },
                args.output / "best.pt",
            )
    (args.output / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved external checkpoint: {args.output / 'best.pt'}")


if __name__ == "__main__":
    main()
