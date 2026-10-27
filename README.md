# ForgetCheck

A local kit that checks whether a model really forgot something.

## Pitch

Teams that unlearn a fact usually check accuracy. Accuracy lies. A model can answer forget facts wrong and still carry a clear trace of them in its losses. ForgetCheck checks three things and reports each one separately. Forget accuracy at chance. A loss based membership test that cannot tell forget facts from facts the model never saw. And retain accuracy staying high. It also trains a gold standard model from scratch on retain facts only, so you can see what real forgetting looks like.

## What it does

It trains a tiny fact model on invented facts. 28 retain facts, 10 forget facts, 10 unseen facts. Then it unlearns the forget facts two ways and verifies both against the gold standard.

Random label unlearning trains forget facts toward a wrong object. Gradient ascent unlearning pushes the loss up on forget facts. Both stop early when forget accuracy reaches chance.

The verifier uses a membership inference AUC. The detector calls a fact a member when its loss is low. AUC 1.0 means forget facts are still easy to spot. AUC 0.5 means the detector cannot tell them from unseen facts.

## How to run

No API key. No network. No downloads. Python 3.10 or newer.

```
cd forgetcheck
python3 run_demo.py
```

Tests need pytest only:

```
pip install -r requirements.txt
python3 -m pytest tests/ -q
```

The demo writes `runs/report.json`.

## Real numbers from the demo run

Run on 2026-10-26. The whole demo took 3.3 seconds. Chance accuracy is 0.083 with 12 objects.

| Model | Retain acc | Forget acc | Forget loss | Unseen loss | MIA AUC | Verdict |
|---|---|---|---|---|---|---|
| Base, trained on retain plus forget | 1.000 | 1.000 | 0.04 | 6.50 | 1.000 | FAIL |
| Gold, retrained on retain only | 1.000 | 0.100 | 7.07 | 6.42 | 0.420 | PASS |
| Random label unlearning, 8 epochs | 1.000 | 0.200 | 1.42 | 6.53 | 0.890 | FAIL |
| Gradient ascent unlearning, 8 epochs | 0.821 | 0.200 | 9.69 | 10.18 | 0.540 | FAIL |

The finding is the demo. Random label fixes accuracy and keeps retain accuracy at 1.000, but its MIA AUC is 0.890, so the forget facts are still easy to spot from their losses. Gradient ascent fixes the MIA AUC at 0.540, close to the gold standard at 0.420, but retain accuracy drops to 0.821. Neither simple method passes all three checks. Only the gold standard passes.

Tests: 9 passed in 6.4 seconds.

## Honest limits

The model is a toy. It is a small network in plain Python that maps invented subject and relation pairs to invented places. It is not a language model. Nothing here proves anything about forgetting in a large language model. The methods are simple baselines, not new unlearning research. The membership test is loss based only, with 10 forget facts and 10 unseen facts, so the AUC is noisy. The point of this project is the verification method and the comparison against a retrained gold standard, not a claim that either unlearning method works in general.

## Credit

The unlearning methods here are standard baselines. Gradient ascent follows Jang et al., ACL 2023. The gold standard comparison is the honest baseline that unlearning papers use: a model retrained without the forget data. The ripple and collateral damage framing follows RippleBench and PreUnlearn, which measure how targeted changes spread to related knowledge. ForgetCheck does not reimplement those tools. It applies their lesson at toy scale: check the change, check the neighbours, and compare against retraining.
