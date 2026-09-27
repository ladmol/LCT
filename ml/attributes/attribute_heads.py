"""Auxiliary make/model/restyling classifiers for approved external data."""

from __future__ import annotations

from torch import nn


class AttributeHeads(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        make_classes: int,
        model_classes: int,
        restyling_classes: int,
    ):
        super().__init__()
        self.make = nn.Linear(embedding_dim, make_classes)
        self.model = nn.Linear(embedding_dim, model_classes)
        self.restyling = nn.Linear(embedding_dim, restyling_classes)

    def forward(self, embedding):
        return {
            "make": self.make(embedding),
            "model": self.model(embedding),
            "restyling": self.restyling(embedding),
        }
