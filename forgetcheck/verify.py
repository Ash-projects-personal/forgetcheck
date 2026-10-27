"""Verification: the part that matters.

Accuracy alone cannot prove forgetting. A model can answer forget
facts wrong and still carry a clear trace of them in its losses. So
ForgetCheck checks three things, and reports each one separately:

1. Forget accuracy is at chance. With 12 objects, chance is about 0.08.
2. Membership inference AUC is near 0.5. A loss based detector tries
   to tell forget facts apart from unseen facts. AUC 1.0 means the
   forget facts are still easy to spot. AUC 0.5 means the detector
   cannot tell them from facts the model never saw.
3. Retain accuracy stays high. Forgetting is easy if you wreck the
   whole model. The check only passes when the kept facts survive.

The gold standard is a model retrained from scratch on retain facts
only, with the same seed and settings. An unlearned model should look
like that gold model on all three checks.
"""


def mia_auc(forget_losses, unseen_losses):
    """Loss based membership inference AUC.

    The detector calls a fact a member when its loss is low. AUC is
    the share of forget/unseen pairs where the forget loss is lower.
    Ties count as half. 0.5 is chance. 1.0 is perfect detection.
    """
    if not forget_losses or not unseen_losses:
        raise ValueError("both loss lists must be non empty")
    wins = 0.0
    total = 0
    for forget_loss in forget_losses:
        for unseen_loss in unseen_losses:
            total += 1
            if forget_loss < unseen_loss:
                wins += 1.0
            elif forget_loss == unseen_loss:
                wins += 0.5
    return wins / total


def verdict(forget_accuracy, retain_accuracy, auc, chance=1.0 / 12.0):
    """One verdict from the three checks. Returns pass or fail plus reasons."""
    reasons = []
    if forget_accuracy > chance + 0.17:
        reasons.append("forget accuracy %.2f is above chance" % forget_accuracy)
    if not 0.35 <= auc <= 0.65:
        reasons.append("MIA AUC %.3f is far from 0.5" % auc)
    if retain_accuracy < 0.90:
        reasons.append("retain accuracy %.2f is below 0.90" % retain_accuracy)
    return {"passed": not reasons, "reasons": reasons}
