---
layout: distill
title: Grokking Baseline Revisited
description: The ICLR 2025 paper, Grokking at the Edge of Numerical Stability, (Prieto et al., 2025)
  and the invention of Muon optimizer have renewed interest in classic grokking experiments and
  therefore raised the importance of accessing how well the baseline can perform in such experiments.
  Here we report replication failures of some of the experiments in Prieto et al., 2025 and improved
  AdamW MLP baseline on modular addition. With tuned learning rate and weight decay AdamW performs
  surprisingly well and may be hard to improve upon.
date: 2025-07-07
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Jason Chuan-Chih Chou
    affiliations:
      name: Cohere Labs Community

# must be the exact same name as your blogpost
bibliography: 2025-07-07-grokking-baseline-revisited.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Discrepancies of Prieto et al., 2025
  - name: AdamW WD baseline

_styles: >
  r { color: Red }
  g { color: Green }
---

## Introduction
First reported in <d-cite key="power2022grokkinggeneralizationoverfittingsmall"></d-cite>, "grokking"
refers to the phenomenon of extremely delayed generalization during model training, in which the model
reaches near-perfect accuracy on the training set orders of magnitude earlier than on the test set.
More recently, the ICLR 2025 paper "Grokking at the Edge of Numerical Stability" (Prieto et al., 2025, <d-cite key="prieto2025grokking"></d-cite>)
reports improved grokking performance with modified softmax function and AdamW optimizer, and people
have been benchmarking variants of Muon optimizer <d-cite key="jordan2024muon"></d-cite> against the
the AdamW baseline on grokking experiments <d-cite key="tveit2025muonoptimizeracceleratesgrokking"></d-cite><d-cite key="cesista2025spectralclipping"></d-cite><d-cite key="EssentialAI2025muongrokking"></d-cite>.
It is perhaps imperative, therefore, to double-check how well the AdamW baseline can really perform for
these experiments. Following <d-cite key="prieto2025grokking"></d-cite>, all the experiments presented
in this post are on the AdamW MLP baseline for modular addition: $$(a + b)\, \mathrm{mod} \, p $$ where
$$a, b \in [0, p), \, p = 113$$ unless specified otherwise.

## Discrepancies of Prieto et al., 2025
We start by replicating the results of Prieto et al., 2025 based on [the official repo](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability) but encounter
unexpected issues, some of which are likely inadvertent:

1. [Discrepancy between `AlgorithmicDataset` and `AlgorithmicDatasetTransformer`](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/issues/6)

    While intended to generate equivalent datasets with different input format (1-hot embedding vs.
    class index), these two dataset classes are in fact not equivalent. The input of the former are
    in the range $$[0, p)$$:
    ```python
    for x in range(0,self.input_size):
        for y in range(0,self.input_size):
            # (...)
    ```
    While the input of the latter are in the range $$[1, p)$$:
    ```python
    self.data = torch.tensor([(i, j) for i in range(1, p) for j in range(1, p)], dtype=torch.long)
    ```
    This means that the results of the transformer models are not fully comparable to that of the
    MLP models. For $$p = 113$$ the difference is <2% but since zero is the additive identity, the
    impact on the model training and evaluation can be disproportionate.

2. [Fig. 1a experiment](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/issues/7)

    Legend of Fig. 1 indicates that the experiment for Fig. 1a is run with modified AdamW optimizer
    ($$\perp$$AdamW) and modified softmax function (StableMax):

    <div class="caption">
      <img src="{{ 'assets/img/2025-07-07-grokking-baseline-revisited/figure_1_crop.png' | relative_url }}" class="img-fluid" width="auto" height="auto">
    </div>
    The correponding command in the experiment script, however, defaults to the standard softmax function:
    ```bash
    python grokking_experiments.py --lr 0.01 --num_epochs 300 --log_frequency 10 --device "$DEVICE" --train_fraction 0.4 --beta2 0.99 --orthogonal_gradients
    ```
    The model underperforms when the same command is run with `--loss_function stablemax`:
    <div class="caption">
      <img src="{{ 'assets/img/2025-07-07-grokking-baseline-revisited/pAdamW_stablemax.png' | relative_url }}" class="img-fluid" width="50%" height="auto">
    </div>

3. [`betas` and `eps` are never set for $$\perp$$AdamW experiments](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/issues/11)

    $$\perp$$AdamW runs AdamW as the `base_optimizer` after orthogonalizing the gradients, but only LR and WD coefficient are passed for its initialization.
    Consequently, the rest of the hyperparameters always default to the PyTorch default despite obvious intention to change $$\beta_2$$ and $$\epsilon$$.

Others, however, are harder to explain:

1. [Weight-decay (WD) experiments of Fig. 15a](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/issues/8)

    See [next section](#adamw-wd-baseline).

2. [Fig. 2a reproduction issues](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/issues/9)
    <div class="caption">
      <img src="{{ 'assets/img/2025-07-07-grokking-baseline-revisited/figure_2a_crop.png' | relative_url }}" class="img-fluid" width="50%" height="auto">
    </div>
    Fig. 2a above purportedly shows that models trained on 0.4 of the dataset all exhibit "softmax collapse" (SC)
    and reach 50% SC before making any progress on test accuracy. The provided plotting code, however,
    shows that the vertical dotted lines supposedly mark the time when 40% SC is reached:
    ```python
    ax.axvline(torch.where(torch.tensor(softmax_collapse_16>0.4))[0][0]*log_frequency, color=colors[0], linewidth=2.5, linestyle=':', label="50% zero terms in the loss")
    ax.axvline(torch.where(torch.tensor(softmax_collapse_32>0.4))[0][0]*log_frequency, color=colors[1], linewidth=2.5, linestyle=':', label="50% zero terms in the loss")
    ax.axvline(torch.where(torch.tensor(softmax_collapse_64>0.4))[0][0]*log_frequency, color=colors[2], linewidth=2.5, linestyle=':', label="50% zero terms in the loss")
    ```
    Furthermore, inspection of the logged metrics shows that the softmax precision 32 and 64
    experiments never reach 40% and 50% SC, respectively.

3. [Fig. 6 reproduction failure](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/issues/10)

    <div class="caption">
      <img src="{{ 'assets/img/2025-07-07-grokking-baseline-revisited/figure_6_crop.png' | relative_url }}" class="img-fluid" width="auto" height="auto">
    </div>

    Fig. 6b above fails to reproduce with the published code, script, and plotting notebook. In particular,
    $$\perp$$AdamW significantly underperforms the shown result:

    <div class="caption">
      <img src="{{ 'assets/img/2025-07-07-grokking-baseline-revisited/figure_6b_repro.png' | relative_url }}" class="img-fluid" width="50%" height="auto">
    </div>

    Perhaps less critically, the transformer experiments of Fig. 6a are in fact performing modular
    addition instead of subtraction according to the experiment script. These experiments are also
    affected by the [dataset discrepancy](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/issues/6)
    although the qualitative result may not change.

## AdamW WD baseline

For the field as a whole, however, perhaps the most important question is how well the AdamW WD
baseline can perform for these grokking experiments. In Fig. 15 of Prieto et al., 2025, the authors
sweep the WD coefficient and compare the experiments with the best setting with that of $$\perp$$AdamW:
<div class="caption">
  <img src="{{ 'assets/img/2025-07-07-grokking-baseline-revisited/figure_15_crop.png' | relative_url }}" class="img-fluid" width="auto" height="auto">
</div>
While Prieto et al. provide no details for these experiments, WD experiments run with the most
common setup of the main experiments (LR=0.01 and trained with 0.4 of the dataset) significantly
outperform the ones shown in Fig. 15a:

```bash
#!/usr/bin/env bash
DEVICE="cuda:0"
for WD in 2. 4. 6. 8. 10.
do
    python grokking_experiments.py --lr 0.01 --weight_decay $WD --num_epochs 1001 --log_frequency 10 --device $DEVICE --train_fraction 0.4
done
```
<div class="caption">
  <img src="{{ 'assets/img/2025-07-07-grokking-baseline-revisited/figure_15a_repro_zoom_in.png' | relative_url }}" class="img-fluid" width="50%" height="auto">
</div>

Generalization of the WD experiments can be further sped up with LR tuning so that $$\perp$$AdamW
no longer appears favorable in comparison:

<div class="caption">
  <img src="{{ 'assets/img/2025-07-07-grokking-baseline-revisited/best_experiments.png' | relative_url }}" class="img-fluid" width="50%" height="auto">
</div>
