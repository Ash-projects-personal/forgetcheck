"""ForgetCheck: verify that a small model really forgot something.

The model here is a tiny fact model, written in plain Python with no
heavy dependencies. It learns subject and relation pairs and predicts
an object. That is enough to show the verification idea on real runs.
It is not a language model, and this README says so plainly.
"""

from .data import build_dataset, Dataset
from .model import FactModel, evaluate, train_model
from .verify import mia_auc, verdict
from .unlearn import unlearn_gradient_ascent, unlearn_random_label

__all__ = [
    "Dataset",
    "FactModel",
    "build_dataset",
    "evaluate",
    "mia_auc",
    "train_model",
    "unlearn_gradient_ascent",
    "unlearn_random_label",
    "verdict",
]
