"""Synthetic fact data for ForgetCheck.

Facts are invented people, relations, and places. Nothing here is a
real person or a real fact. The data exists only so the verifier has
something real to measure.

Split, with seed 7:
  retain: 28 facts the model keeps
  forget: 10 facts the model must forget
  unseen: 10 facts the model never sees in training
The unseen set is the baseline for the membership test. A model that
really forgot should treat forget facts the way it treats unseen ones.
"""

import random
from dataclasses import dataclass, field


@dataclass
class Fact:
    subject: str
    relation: str
    obj: str
    subject_id: int
    relation_id: int
    object_id: int


@dataclass
class Dataset:
    facts: list = field(default_factory=list)
    retain: list = field(default_factory=list)
    forget: list = field(default_factory=list)
    unseen: list = field(default_factory=list)
    subjects: list = field(default_factory=list)
    relations: list = field(default_factory=list)
    objects: list = field(default_factory=list)


def build_dataset(seed=7, n_forget=10, n_unseen=10):
    rng = random.Random(seed)
    subjects = ["person_%02d" % i for i in range(12)]
    relations = ["lives_in", "works_at", "likes", "owns"]
    objects = ["place_%02d" % i for i in range(12)]
    facts = []
    for si, subject in enumerate(subjects):
        for ri, relation in enumerate(relations):
            oi = rng.randrange(len(objects))
            facts.append(
                Fact(
                    subject=subject,
                    relation=relation,
                    obj=objects[oi],
                    subject_id=si,
                    relation_id=ri,
                    object_id=oi,
                )
            )
    rng.shuffle(facts)
    forget = facts[:n_forget]
    unseen = facts[n_forget : n_forget + n_unseen]
    retain = facts[n_forget + n_unseen :]
    return Dataset(
        facts=facts,
        retain=retain,
        forget=forget,
        unseen=unseen,
        subjects=subjects,
        relations=relations,
        objects=objects,
    )
