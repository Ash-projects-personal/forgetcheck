"""A tiny fact model in plain Python.

Input is a one hot subject plus a one hot relation. The model is one
hidden layer with tanh, then a softmax over objects. Training is full
batch gradient descent, written out by hand so the whole thing runs
with no deep learning library installed.

This is a toy. It proves the verification method, not a claim about
large language models. The README states that limit in plain words.
"""

import math
import random


class FactModel:
    def __init__(self, n_subjects, n_relations, n_objects, hidden=24, seed=0):
        rng = random.Random(seed)
        self.n_subjects = n_subjects
        self.n_relations = n_relations
        self.n_objects = n_objects
        self.hidden = hidden
        n_in = n_subjects + n_relations
        self.w1 = [[rng.gauss(0, 0.5) for _ in range(hidden)] for _ in range(n_in)]
        self.b1 = [0.0] * hidden
        self.w2 = [[rng.gauss(0, 0.5) for _ in range(n_objects)] for _ in range(hidden)]
        self.b2 = [0.0] * n_objects

    def _input(self, fact):
        x = [0.0] * (self.n_subjects + self.n_relations)
        x[fact.subject_id] = 1.0
        x[self.n_subjects + fact.relation_id] = 1.0
        return x

    def _forward(self, x):
        hidden_pre = [
            sum(x[i] * self.w1[i][j] for i in range(len(x))) + self.b1[j]
            for j in range(self.hidden)
        ]
        hidden = [math.tanh(v) for v in hidden_pre]
        logits = [
            sum(hidden[j] * self.w2[j][k] for j in range(self.hidden)) + self.b2[k]
            for k in range(self.n_objects)
        ]
        top = max(logits)
        exps = [math.exp(v - top) for v in logits]
        total = sum(exps)
        probs = [e / total for e in exps]
        return probs, hidden

    def predict(self, fact):
        probs, _ = self._forward(self._input(fact))
        return max(range(self.n_objects), key=lambda k: probs[k])

    def loss(self, fact):
        probs, _ = self._forward(self._input(fact))
        return -math.log(max(probs[fact.object_id], 1e-12))

    def _gradients(self, facts, sign):
        """Accumulate gradients. Sign 1 descends the loss, sign -1 ascends it."""
        n_in = self.n_subjects + self.n_relations
        gw1 = [[0.0] * self.hidden for _ in range(n_in)]
        gb1 = [0.0] * self.hidden
        gw2 = [[0.0] * self.n_objects for _ in range(self.hidden)]
        gb2 = [0.0] * self.n_objects
        for fact in facts:
            x = self._input(fact)
            probs, hidden = self._forward(x)
            delta = probs[:]
            delta[fact.object_id] -= 1.0
            for j in range(self.hidden):
                for k in range(self.n_objects):
                    gw2[j][k] += sign * delta[k] * hidden[j]
            for k in range(self.n_objects):
                gb2[k] += sign * delta[k]
            d_hidden = [
                sum(delta[k] * self.w2[j][k] for k in range(self.n_objects))
                * (1.0 - hidden[j] * hidden[j])
                for j in range(self.hidden)
            ]
            for i in range(n_in):
                for j in range(self.hidden):
                    gw1[i][j] += sign * d_hidden[j] * x[i]
            for j in range(self.hidden):
                gb1[j] += sign * d_hidden[j]
        return gw1, gb1, gw2, gb2

    def _apply(self, grads, lr, scale):
        gw1, gb1, gw2, gb2 = grads
        for i in range(len(self.w1)):
            for j in range(self.hidden):
                self.w1[i][j] -= lr * scale * gw1[i][j]
        for j in range(self.hidden):
            self.b1[j] -= lr * scale * gb1[j]
        for j in range(self.hidden):
            for k in range(self.n_objects):
                self.w2[j][k] -= lr * scale * gw2[j][k]
        for k in range(self.n_objects):
            self.b2[k] -= lr * scale * gb2[k]


def train_model(model, facts, epochs=200, lr=1.0):
    """Plain training. Descend the loss on the given facts."""
    for _ in range(epochs):
        grads = model._gradients(facts, sign=1.0)
        model._apply(grads, lr=lr, scale=1.0 / len(facts))
    return model


def evaluate(model, facts):
    """Return mean loss and accuracy on a set of facts."""
    if not facts:
        return {"count": 0, "mean_loss": 0.0, "accuracy": 0.0, "losses": []}
    losses = [model.loss(fact) for fact in facts]
    correct = sum(1 for fact in facts if model.predict(fact) == fact.object_id)
    return {
        "count": len(facts),
        "mean_loss": sum(losses) / len(losses),
        "accuracy": correct / len(facts),
        "losses": losses,
    }
