"""Two simple unlearning methods, for comparison.

Random label: train forget facts toward a wrong object, while still
training retain facts normally. This is a known baseline. It usually
fixes accuracy fast and keeps retain accuracy high.

Gradient ascent: push the loss up on forget facts, while still
training retain facts normally. This usually moves the losses, but
it is harder to control and it can damage retain accuracy.

Both stop early, as soon as forget accuracy reaches chance. The
point of the demo is that neither method is a free win, and the
verifier shows which check each one fails.
"""

import random

from .model import evaluate


def _stop_at_chance(model, forget, max_epochs):
    for epoch in range(1, max_epochs + 1):
        yield epoch
        if evaluate(model, forget)["accuracy"] <= 0.20:
            break


def unlearn_random_label(model, retain, forget, epochs=40, lr=1.0, seed=99):
    rng = random.Random(seed)
    mislabeled = []
    for fact in forget:
        wrong = rng.randrange(model.n_objects)
        while wrong == fact.object_id:
            wrong = rng.randrange(model.n_objects)
        changed = type(fact)(
            subject=fact.subject,
            relation=fact.relation,
            obj=fact.obj,
            subject_id=fact.subject_id,
            relation_id=fact.relation_id,
            object_id=wrong,
        )
        mislabeled.append(changed)
    steps = 0
    for epoch in _stop_at_chance(model, forget, epochs):
        steps = epoch
        grads = model._gradients(retain + mislabeled, sign=1.0)
        model._apply(grads, lr=lr, scale=1.0 / len(retain + mislabeled))
    return {"method": "random_label", "epochs_used": steps}


def unlearn_gradient_ascent(model, retain, forget, epochs=20, lr=3.0):
    steps = 0
    for epoch in _stop_at_chance(model, forget, epochs):
        steps = epoch
        grads_retain = model._gradients(retain, sign=1.0)
        grads_forget = model._gradients(forget, sign=-1.0)
        combined = (
            [
                [a + b for a, b in zip(row_a, row_b)]
                for row_a, row_b in zip(grads_retain[0], grads_forget[0])
            ],
            [a + b for a, b in zip(grads_retain[1], grads_forget[1])],
            [
                [a + b for a, b in zip(row_a, row_b)]
                for row_a, row_b in zip(grads_retain[2], grads_forget[2])
            ],
            [a + b for a, b in zip(grads_retain[3], grads_forget[3])],
        )
        model._apply(combined, lr=lr, scale=1.0 / len(retain))
    return {"method": "gradient_ascent", "epochs_used": steps}
