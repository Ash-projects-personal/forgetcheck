import copy

import pytest

from forgetcheck.data import build_dataset
from forgetcheck.model import FactModel, evaluate, train_model
from forgetcheck.unlearn import unlearn_gradient_ascent, unlearn_random_label
from forgetcheck.verify import mia_auc, verdict


def fresh(data):
    return FactModel(
        n_subjects=len(data.subjects),
        n_relations=len(data.relations),
        n_objects=len(data.objects),
        hidden=24,
        seed=0,
    )


def trained(data):
    model = fresh(data)
    train_model(model, data.retain + data.forget, epochs=200, lr=1.0)
    return model


def test_splits_are_disjoint_and_cover_all_facts():
    data = build_dataset()
    keys = lambda facts: {(f.subject_id, f.relation_id) for f in facts}
    assert len(data.retain) == 28
    assert len(data.forget) == 10
    assert len(data.unseen) == 10
    assert not (keys(data.retain) & keys(data.forget))
    assert not (keys(data.retain) & keys(data.unseen))
    assert not (keys(data.forget) & keys(data.unseen))
    assert len(keys(data.facts)) == 48


def test_mia_auc_is_one_for_perfect_separation():
    assert mia_auc([0.1, 0.2], [5.0, 6.0]) == 1.0


def test_mia_auc_is_half_for_identical_losses():
    assert mia_auc([1.0, 2.0], [1.0, 2.0]) == 0.5


def test_mia_auc_rejects_empty_lists():
    with pytest.raises(ValueError):
        mia_auc([], [1.0])


def test_verdict_passes_only_when_all_three_checks_hold():
    assert verdict(0.10, 1.00, 0.50)["passed"] is True
    assert verdict(1.00, 1.00, 0.50)["passed"] is False
    assert verdict(0.10, 1.00, 1.00)["passed"] is False
    assert verdict(0.10, 0.50, 0.50)["passed"] is False


def test_training_memorizes_retain_and_forget():
    data = build_dataset()
    model = trained(data)
    assert evaluate(model, data.retain)["accuracy"] == 1.0
    assert evaluate(model, data.forget)["accuracy"] == 1.0
    assert evaluate(model, data.unseen)["accuracy"] <= 0.2


def test_gold_standard_looks_like_it_never_saw_forget_facts():
    data = build_dataset()
    gold = fresh(data)
    train_model(gold, data.retain, epochs=200, lr=1.0)
    forget = evaluate(gold, data.forget)
    unseen = evaluate(gold, data.unseen)
    assert forget["accuracy"] <= 0.2
    assert mia_auc(forget["losses"], unseen["losses"]) < 0.65


def test_random_label_fixes_accuracy_but_not_mia():
    data = build_dataset()
    model = trained(data)
    other = copy.deepcopy(model)
    unlearn_random_label(other, data.retain, data.forget)
    forget = evaluate(other, data.forget)
    unseen = evaluate(other, data.unseen)
    assert forget["accuracy"] <= 0.2
    assert evaluate(other, data.retain)["accuracy"] == 1.0
    assert mia_auc(forget["losses"], unseen["losses"]) > 0.65


def test_gradient_ascent_fixes_mia_but_costs_retain():
    data = build_dataset()
    model = trained(data)
    other = copy.deepcopy(model)
    unlearn_gradient_ascent(other, data.retain, data.forget)
    forget = evaluate(other, data.forget)
    unseen = evaluate(other, data.unseen)
    assert forget["accuracy"] <= 0.2
    assert mia_auc(forget["losses"], unseen["losses"]) < 0.65
    assert evaluate(other, data.retain)["accuracy"] < 1.0
