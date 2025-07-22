---
layout: distill
title: Grokking Baseline Revisited
description: The ICLR 2025 paper, Grokking at the Edge of Numerical Stability, (Prieto et al., 2025)
  and the invention of Muon optimizer have renewed interest in classic grokking experiments and
  therefore raised the importance of accessing how well the baseline can perform in such experiments.
  Here we report replication failures of some of the experiments in Prieto et al., 2025 and improved
  AdamW MLP baseline on modular addition. With tuned learning rate and weight decay AdamW performs
  surprisingly well and may be hard to improve upon.
date: 2025-07-21
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Jason Chuan-Chih Chou
    affiliations:
      name: Cohere Labs Community

# must be the exact same name as your blogpost
bibliography: 2025-07-21-grokking-baseline-revisited.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Discrepancies of Prieto et al., 2025
  - name: AdamW WD baseline
  - name: Lipschitz measurements
  - name: Conclusions
  - name: Replication guide
    subsections:
    - name: AdamW WD baseline
    - name: Lipschitz measurements

_styles: >
  r { color: Red }
  g { color: Green }
  gr { color: Gray }
---

## Introduction
First reported in <d-cite key="power2022grokkinggeneralizationoverfittingsmall"></d-cite>, "grokking"
refers to the phenomenon of extremely delayed generalization during model training, in which the model
reaches near-perfect accuracy on the training set orders of magnitude earlier than on the test set.
More recently, the ICLR 2025 paper "Grokking at the Edge of Numerical Stability" (Prieto et al., 2025, <d-cite key="prieto2025grokking"></d-cite>)
reports improved grokking performance with modified softmax function and AdamW optimizer <d-cite key="loshchilov2018decoupled"></d-cite>, and people
have been benchmarking variants of Muon optimizer <d-cite key="jordan2024muon"></d-cite> against the
AdamW baseline on grokking experiments <d-cite key="tveit2025muonoptimizeracceleratesgrokking"></d-cite><d-cite key="cesista2025spectralclipping"></d-cite><d-cite key="EssentialAI2025muongrokking"></d-cite>.
It is perhaps imperative, therefore, to double-check how well the AdamW baseline can really perform for
these experiments. Following <d-cite key="prieto2025grokking"></d-cite>, all the experiments presented
in this post are on the AdamW baseline from the paper with a 2-hidden layer multi-layer perceptron
(MLP) of width 200 for modular addition: $$(a + b)\, \mathrm{mod} \, p $$ where
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
      <img src="{{ 'assets/img/2025-07-21-grokking-baseline-revisited/figure_1_crop.png' | relative_url }}" class="img-fluid" width="auto" height="auto">
    </div>
    The correponding command in the experiment script, however, defaults to the standard softmax function:
    ```bash
    python grokking_experiments.py --lr 0.01 --num_epochs 300 --log_frequency 10 --device "$DEVICE" --train_fraction 0.4 --beta2 0.99 --orthogonal_gradients
    ```
    The model underperforms when the same command is run with `--loss_function stablemax`:
    <div class="caption">
      <img src="{{ 'assets/img/2025-07-21-grokking-baseline-revisited/pAdamW_stablemax.png' | relative_url }}" class="img-fluid" width="50%" height="auto">
    </div>

3. [`betas` and `eps` are never set for $$\perp$$AdamW experiments](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/issues/11)

    $$\perp$$AdamW runs AdamW as the `base_optimizer` after orthogonalizing the gradients, but only LR and WD (weight decay) coefficient are passed for its initialization.
    Consequently, the rest of the hyperparameters always default to the PyTorch default despite obvious intention to change $$\beta_2$$ and $$\epsilon$$.

Others, however, are harder to explain:

1. [Weight-decay (WD) experiments of Fig. 15a](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/issues/8)

    See [next section](#adamw-wd-baseline).

2. [Fig. 2a reproduction issues](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/issues/9)
    <div class="caption">
      <img src="{{ 'assets/img/2025-07-21-grokking-baseline-revisited/figure_2a_crop.png' | relative_url }}" class="img-fluid" width="50%" height="auto">
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
      <img src="{{ 'assets/img/2025-07-21-grokking-baseline-revisited/figure_6_crop.png' | relative_url }}" class="img-fluid" width="auto" height="auto">
    </div>

    Fig. 6b above fails to reproduce with the published code, script, and plotting notebook. In particular,
    $$\perp$$AdamW significantly underperforms the shown result:

    <div class="caption">
      <img src="{{ 'assets/img/2025-07-21-grokking-baseline-revisited/figure_6b_repro.png' | relative_url }}" class="img-fluid" width="50%" height="auto">
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
  <img src="{{ 'assets/img/2025-07-21-grokking-baseline-revisited/figure_15_crop.png' | relative_url }}" class="img-fluid" width="auto" height="auto">
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
  <img src="{{ 'assets/img/2025-07-21-grokking-baseline-revisited/figure_15a_repro_zoom_in.png' | relative_url }}" class="img-fluid" width="50%" height="auto">
</div>

Generalization of the WD experiments can be further sped up with LR tuning so that $$\perp$$AdamW
no longer appears favorable:

<div class="caption">
  <img src="{{ 'assets/img/2025-07-21-grokking-baseline-revisited/best_experiments.png' | relative_url }}" class="img-fluid" width="50%" height="auto">
</div>

For a more systematic comparison, we test architecture
variations along two axes<d-footnote>Prieto et al.’s MLP 2-hot encodes the operands of modular addition $$a, b \in [0, p)$$ by
concatenating their 1-hot encoding. Prieto et al. claim that this MLP is
from <d-cite key="liu2023omnigrok"></d-cite>, which in turn claims that the setup is from
Liu et al., 2022 <d-cite key="liu2022towards"></d-cite>. Liu et al.'s repository, however,
shows that <a href="https://github.com/ejmichaud/grokking-squared/blob/0229df94de69b8384e560367280a43a238112bf5/toy/train_add.py#L31-L33">its toy modular addition model first adds up the embeddings of a, b before feeding the sum to the 2-hidden layer MLP</a>. In contrast, Prieto et al.’s MLP feeds the 2-hot encoding directly to the MLP so its first hidden
layer effectively adds up the "embeddings" of a, b and the overall model has one less hidden
layer, in addition of the difference of nonlinearity (tanh vs. ReLU) and the fact that Prieto et al.'s
MLP applies ReLU to the sum of "embeddings" first. The architecture variations we test are motivated
by this discrepancy and the dimension of the trainable embeddings is set to 100 to keep the dimension
of the concatenated embedding the same as that of the hidden layers (200).</d-footnote>:

1. Number of hidden layers $$\in \{1, 2, 3\}$$
2. Trainable embedding layer instead of 2-hot encoding for the operands $$a, b$$

For each variation we grid-search LR $$\in \{0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1\}$$
for both the $$\perp$$AdamW and WD experiments. For the WD experiments, we also grid-search the WD
coefficient $$\in \{2, 4, 6, 8, 10\}$$. We then compare the first epoch to solve modular addition
with $$p = 113$$ (the first epoch that the model reaches 100% accuracy on the test set) between the
best WD experiment and the best $$\perp$$AdamW experiment for each architecture variation.
The difference is marked <g>green</g> if the best WD experiment solves first and <r>red</r> if the
best $$\perp$$AdamW experiment solves first. We do not train beyond 400 epochs for the 3 hidden
layers experiments since smaller models can solve modular addition within half of the training
budget:

||2-hot Encoding|Embedding Layer|
|:-----:|:-----:|:-----:|
|1 Hidden Layer|917 <g>-119</g>|126 <g>-140</g>|
|2 Hidden Layers|143 <r>+19</r>|111 <g>-87</g>|
|3 Hidden Layers|>400 <gr>±?</gr>|146 <g>->254</g>|

We can see that $$\perp$$AdamW doesn't hold a systematic advantage over the simple WD baseline. In fact,
it underperforms across the board with more conventional trainable embedding. We have also tested setting
WD to 0 for the embedding layer and running the PyTorch default $$\beta_2$$ and $$\epsilon$$ for $$\perp$$AdamW
like the original repo, but neither helps.

## Lipschitz measurements

Cesista, 2025 <d-cite key="cesista2025spectralclipping"></d-cite> has experimented on a variant of the MLP model and measured
how fast variants of the Muon optimizer can train the model to grok vs. how robust the models are. How
does the AdamW WD baseline compare?

To recap, one way to measure the model's robustness is to measure the Lipschitz constant $L$:

> **Definition (Lipschitz)**. Let $f: \mathbb{R}^n \to \mathbb{R}^m$ be a function, then $f$ is said to be $L$-Lipschitz continuous if there exists a constant $L \geq 0$ such that for all $x, y \in \mathbb{R}^n$,
$$||f(x) - f(y)|| \leq L||x - y||$$
> for some norm $||\cdot||$ chosen a priori.

Smaller Lipschitz constant $L$ means that the model is less sensitive to input perturbation, therefore
can be considered more robust. In our case of a 2-hidden layer MLP that takes the concatenation of
two 113-dim embeddings as the input and outputs the 113-dim logit, $n = 226$ and $m = 113$. With L2
norm $||\cdot||_2$ as the norm of choice and 1-Lipschitz functions such as ReLU or GELU as the
activation function, the product of the spectral norms of the 3 linear layers constitutes a upper
bound of the Lipschitz constant $L$ where the spectral norm of a linear transformation $M$ is its
largest singular value:

$$||M||_2 = \max_{x \neq 0} \frac{||Mx||_2}{||x||_2}$$

which can efficiently approximated with [power iteration](https://en.wikipedia.org/wiki/Power_iteration).
Following Cesista, 2025 <d-cite key="cesista2025spectralclipping"></d-cite>, we train such MLP with
GELU activation function, AdamW, and bfloat16 precision to grok modular addition and modular
multiplication with $p=113$. We again grid-search LR $$\in \{0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1\}$$
and the WD coefficient $$\in \{2, 4, 6, 8, 10\}$$ and run 64 random seeds for each setting. We filter
out settings in which the model doesn't reach 95% test accuracy 100% of the time within 1000 epochs
(equivalent to 1000 steps with full-batch training), and report the median epochs to grok and the
median Lipschitz upper bound of the remaining settings:

<div class="caption">
  <img src="{{ 'assets/img/2025-07-21-grokking-baseline-revisited/gelu_epoch_lipschitz.png' | relative_url }}" class="img-fluid" width="50%" height="auto">
</div>

Compared to the results reported in Cesista, 2025 <d-cite key="cesista2025spectralclipping"></d-cite>,
AdamW breaks the Pareto frontier of Muon variants with models trained to grok within 200 epochs and
Lipschitz upper bound < 200. We don't see clear correlation between epochs to grok and the model's
Lipschitz upper bound. Switching from GELU to ReLU doesn't change the result significantly.

## Conclusions

Inline with <d-cite key="EssentialAI2025muongrokking"></d-cite>, we find that AdamW with tuned LR and
WD coefficient to be a strong baseline for the task of modular addition and multiplication in comparison
to modified $$\perp$$AdamW and variants of Muon. In fact, with the tasks solved within 150 epochs,
perhaps we should question whether such training dynamics should still be called "grokking"
and whether other common grokking tasks can be similarly solved efficiently with tuned LR and WD. To
the extent that it is still relevant, the AdamW baseline for grokking may be hard to improve upon
significantly.

## Replication guide

### AdamW WD baseline

1. `git clone -b talon https://github.com/EIFY/grokking-at-the-edge-of-numerical-stability.git`
2. `cd grokking-at-the-edge-of-numerical-stability`
3. `./wd_experiments.sh`
4. Run jupyter notebook `wd_plots.ipynb`

### Lipschitz measurements

1. `git clone -b lipschitz https://github.com/EIFY/grokking-at-the-edge-of-numerical-stability.git`
2. `cd grokking-at-the-edge-of-numerical-stability`
3. `./lipschitz_experiments.sh # This takes hours due to running 64 random seeds for each setting`
4. Run jupyter notebook `lipschitz_plots.ipynb`
