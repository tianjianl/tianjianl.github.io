# Manipulating Gradients to Balance Multilingual Machine Translation Models

## Overview

Finding out the reasons and solutions for negative interference in Multilingual Neural Machine Translation [[Johnson et al., 2016](https://arxiv.org/abs/1611.04558); [Aharoni et al., 2019](https://aclanthology.org/N19-1388)] has been an active research area for the past 5-7 years. Yet, while previous studies [[Wang et al., 2020](https://arxiv.org/abs/2010.05874)] finds that negative interference mostly occurs between different langauge familiies, recent studies [[Shaham et al., 2023](https://arxiv.org/abs/2212.07530)] have demonstrated that negative inference is does not occur between languages of different families, and the interference emerges because of the mismatch in the amount of data for different translation directions. Real world translation data suffers from heavy mismatch of data in different directions: ranging from less than 100K to over 100M [[NLLB Team, 2022](https://arxiv.org/abs/2207.04672)], so it is crucial to find balancing methods that are both scalable and robust.

The goal of this blog post is to give an overview on the three approaches for handling imbalances in multilingual neural machine translation: **Scalarization**, **Gradient projection, and Distillation**. Each method has its pros and cons and there is still no consensus on which method performs the best. I will cover some of the most up-to-date methods for handling datasize mismatches and interference in this blog post. 

Disclaimer: The papers introduced in this blogpost are only representative works rather than a comprehensive survey to give an overview of what are the different methods to handle data imbalance in translation directions. 

## Background

#### Multilingual Neural Machine Translation

We start by describing some basics in multilingual neural machine translation: we are interested in mapping a source sequence $\textbf{x}_s = \{x_1, x_2, ..., x_n\}$ in language $s$ to a target sequence $\textbf{y}_t = \{y_1, y_2, ..., y_m\}$ in language $t$. We train an autoregressive model parameterized by $\theta$ that predicts each target token conditioning on the entire source sentence and the target tokens before it:

$$ \mathcal{L}_{s,t}(\theta) = \sum_{i=1}^m \log p_\theta(y_i \mid \mathbf{y}_{<i}, \mathbf{x}).$$

In multilingual neural machine translation, there are multiple source-target pairs concatenated together to form a large dataset. Given parallel sentences in $N$ languages pairs $(s_1, t_1), (s_2, t_2),... (s_N, t_N)$ , a naive multilingual machine translation model aims to minimize an unweighted sum of the losses of individual translation directions: 

$$\mathcal{L}_\text{MMT}(\theta) = \sum_{i=1}^N \mathcal{L}_{s_i, t_i}(\theta)$$

Here the parameters for each translation direction are shared, which is much more compute-efficient compared to training individual models for each direction and different directions might benefit from each other, resulting in positive transfer.

#### Pareto Front

Multilingual Translation can be seen as a multi-task learning (or multi-objective optimizaiton) problem [[Ahroni et al., 2019](https://aclanthology.org/N19-1388/)], where each translation direction is an individual task. We are interested in finding the optimal solutions, optimal in that we cannot improve the performance of an individual task without sacrificing the performance of other tasks. Such solutions are called **Pareto optimal solutions** [[Boyd and Vandenberghe, 2004](https://web.stanford.edu/~boyd/cvxbook/)]. The set of all Pareto optimal solutions forms the **Pareto Front** of a given multi-objective optimization problem. See figure 1 for an illustration of the pareto front (figure pasted from [Fernandes et al., 2023](https://arxiv.org/abs/2302.09650)).

{% include figure.html path="assets/img/Screenshot%202023-11-08%20at%2011.43.59 PM.png" class="img-fluid rounded z-depth-1" zoomable=true width="391"%}

## Scalarization

The heavy mismatch in datasizes causes the naive unweighted average of individual losses amplifies the performance on high-resource languages (HRLs) and deceases the performance on low resource languages (LRLs) In practice, we sample a batch of input and output sentences, and since data of HRLs can be more than 10x of data of LRLs, we are far more likely to sample from HRLs, resulting in the optimization process heavily favoring optimizing towards better performance on HRLs. 

To mitigate this, instead of using porportional sampling, we can use temperature sampling [[Arivazhagan et al., 2019](https://arxiv.org/abs/1907.05019)]. The probability of sampling from each direction is given by:

$$
p_{(s_i, t_i)} = \frac{D(s_i, t_i)^\frac{1}{\tau}}{\sum_{j=1}^N D(s_j, t_j)^\frac{1}{\tau}}
$$

Where $D(s_i, t_i)$ is the datasize of translation direction $s_i \rightarrow t_i$ , and $\tau$ is the sampling temperature. When $\tau = 1$, our sampling method is equivalent to naive porportionaling sampling. As $\tau$  gets larger, we are decreasing the weights on HRLs and increasing the weights on LRLs. As $\tau \rightarrow +\infty$, the sampling strategy becomes a uniform distribution over all language pairs. 

A common understanding in the machine tranlation literature is that tuning the temperature $\tau$ is equivalent to tuning the weights for each translation direction in a weighted sum of individual losses, which we refer to as **scalariztion**. 

$$
\mathcal{L}_\text{MMT}(\theta) = \sum_{i=1}^N w_i \mathcal{L}_{s_i, t_i}(\theta)
$$

Although the equivalency of tuning the temperature $\tau$ and tuning the weights $w_i$ have not been throughly established, in this blog post, I will use **sampling ratio** and **task weights** interchangably and introduce some of the latest advances of how to find fantanstic weights that achieves good performance.

#### Static Weights

[Fernandes et al., 2023](https://arxiv.org/abs/2302.09650) shows that we can find the pareto front for multilingual translation by varying the sampling ratio. However, their work assumes that we have an even amount of data for each translation direction, which is often not the case in real-world settings. A follow up work [[Chen et al., 2023](https://arxiv.org/abs/2304.03216)] shows that when there is a data size mismatch, the pareto front collapses: see the following figure from the paper for an comparison between the pareto curve when data is balanced V.S. data is imbalanced.

{% include figure.html path="assets/img/Screenshot%202023-11-08%20at%204.45.32 PM.png" class="img-fluid rounded z-depth-1" width="602" zoomable=true %}

presents a comprehensive study on how to tune the sampling temperature (or equivalently, task weights) for each translation direction. Given a fixed sampling ratio $p$ and the number of training examples $D$ for a given direction, the cross-entropy loss can be expressed as:

$$
\mathcal{L}(p, D) = (k\cdot p)^{-\alpha} + (D^\gamma + b) \cdot(q\cdot p)^{\beta}+M_{\infty}
$$

The given parameters are the sampling ratio $p$, the data size $D$ and a constant bias term $M_{\infty}$, and the rest of the parameters are to be estimated. To estimate these parameters, the authors conducted a series of experiments on WMT  $\text{English} \rightarrow \{\text{French, German}\}$ data and vary the amount of available data $D$ and sampling ratio $p$ to form a curve and estimate parameters that best fit this curve. Experiments show that their estimated scaling laws generalizes to other language pairs as well.

So the question remains: how to find the best $p$ for each direction? The author frame this as an optimization problem: given a fixed bumber of $D$ and a pre-defined importance of each translation direction $r$ for each task, the optimization problem is:

$$
\mathbf{p} = \arg\min_p \mathcal{L}(p;r;d) \\
\textrm{subject to}~\mathcal{L(\mathbf{p};r;d}) = \sum_{i}r_i\mathcal{L}(p_i, D_i)\\
\mathbf{p}>0 \\
\sum_{i} p_i =1,~~~\sum_{i} r_i =1
$$

The standard way is to assume each translation direction has equal importance $r_i = \frac{1}{N}$ , but this is customizable if you want to emphasize some translation directions.  Such a static weighting method for multilingual translation produces strong baselines over using a fixed temperature (e.g. $\tau = 1, 5, 100$) and fancy gradient manipulation techniques (which we will introduce later). 

#### Dynamic Weights

But why do we need static weights for each direction? The mismatch in data sizes causes mismatch in convergence rates, as [Huang et al., 2022](https://arxiv.org/abs/2205.01620) points out, while LRLs have already converged, the HRLs have not yet converged, and continuing training on all tasks results in overfitting on the LRLs. This motivates methods that tackle the imbalanced problem with dynamic sampling temperature methods that takes into account that different directions converge at a different rate. 

A naive way is to focus on one or a set of translation direction that converges the slowest during training: In statistical learning, Distributionall Robust Optimization methods [[Oren et al., 2019](https://arxiv.org/abs/1909.02060); [Sagawa et al., 2020](https://openreview.net/forum?id=ryxGuJrFvS); [Zhou et al., 2021](https://arxiv.org/abs/2109.04020)], instead of minimizing the sum of all losses, tries to minimize the loss of the worst performing group, forming a minmax optimization problem:

$$
\min_\theta \max_{s, t} \mathcal{L_{s ,t}(\theta)}
$$

However, naively minimizing the worse performing langauge pair ignores the fact that several translation direction might have similar datasizes, and thus have similar bad performance, so instead of only focusing on the one worse performing direction, both [Oren et al., 2019](https://arxiv.org/abs/1909.02060) and [Zhou et al., 2021](https://arxiv.org/abs/2109.04020) proposes to minimize the loss of a **set** of worse performing domains/languages. 

Specifically, [Oren et al., 2019](https://arxiv.org/abs/1909.02060) minimizes a fixed **fraction** of worse performing domains in general language modeling.

[Zhou et al., 2021](https://arxiv.org/abs/2109.04020) minimizes the worse case weighted average loss of all language pairs, where the weights/sampling ratios are close to porpotional sampling. More intuitively, the author first finds some adversarial weights that are close to porpotional weights but yields in the worst possible loss of all the weights that are close and aims to minimize this loss. Again, you can be creative in how to define closeness for two probability distributions but in their work, they used the $\chi$- Divergence because it has some nice properties [[Duchi and Namkoong, 2016](https://arxiv.org/abs/1610.02581); [Hashimoto et al., 2018](https://arxiv.org/abs/1806.08010)]. 

Experiment results show that maximizing the worse case loss can improve the performance on LRLs while minimally sacrificing the performance on HRLs, essentially pushing forward the pareto frontier. 

[Li and Gong, 2021](https://openreview.net/forum?id=Rv3vp-JDUSJ) find weights for each direction that guide the optimization process towards a flatter minima. The improvements are larger compared to our previously introduced Distributionally Robust Optimization works, which highlight that the optimization process for multlingual translation should take the differences in convergence (or in this case, curvature) into account. 

We can also make the weights **learnable**, depending on how to measure how **well** is the training process going, we bias our sampling ratio towards training regimes that are **well**. 

For example, we can select sampling ratio so that our training gradient is most similar to development gradient [[Wang et al., 2020](https://arxiv.org/abs/2004.06748)].

We can also select sampling ratio to minimize loss related definitions of learning progess. [[Kreutzer et al., 2021](https://arxiv.org/abs/2109.04020)].

<u>But, perhaps the simplest of all methods is more robust and generalizable</u>: 

Instead of searching for these fantanstic weights, we don't we first train on HRLs and then train on a mixture of HRLs and LRLs? Well, it turns out this simple method works surprizingly well: 

[Choi et al., 2023](https://openreview.net/forum?id=7RMGI4slcb) proposes to first train on HRLs, then "fine-tune" on a mixture of HRLs and LRLs, which is equivalent to tuning the temperature but in a more coarse-grained way. Instead of using the training signals (gradient, activations), this work manually divides the training into two stages - training on HRLs first and a mix of high and low reource languages second. This simple trick solves the mismatch in convergence that causes overfitting on the LRLs.

## Gradient Manipulation

It was not until recently [[Xin et al., 2022](https://arxiv.org/abs/2209.11379)] that we realize we don't need fancy techniques designed for multi-task learning to solve unbalanced training in multilingual translation. Simple scalarization often yields strong baselines that are tough to beat. However, there has been extensive studies on how to manipulate the gradients in general multi-task learning setups, and people have been trying them on multilingual machine translation.  

It is natural to assume that the gradient conflict between different tasks causes interference between them. Therefore, prior research have developed methods to either drop some of the conflicting gradients [[Chen et al., 2020](https://arxiv.org/abs/2010.06808)], project one gradient to the orthogonal plane of another [[Yu et al., 2020](https://arxiv.org/abs/2001.06782?context=cs.RO); [Yang et al., 2021](https://arxiv.org/abs/2109.04778)], or taking language similarity into account and project one gradient to a plane where the cosine similarity between gradients reflect language similarity [[Wang et al., 2020](https://arxiv.org/abs/2010.05874)]. See the following figure for an illustration of these methods (red is task 1 and blue is task 2):

{% include figure.html path="assets/img/Screenshot%202023-11-08%20at%2011.45.54 PM.png" class="img-fluid rounded z-depth-1" width="564" zoomable=true %}

As promising as these methods might seem, there has been compelling evidence recently that gradient deconfliction does not outperform simple static scalarization both in the general multi-task learning setting [[Xin et al., 2022](https://arxiv.org/abs/2209.11379); [Kurin et al., 2022](https://arxiv.org/abs/2201.04122)] and specifically for machine translation [[Chen et al., 2023](https://arxiv.org/abs/2304.03216)]. 

## Discussion and Future Work

As you can see, a lot of the papers introduced here are from the past one or two years, so the whole area of how to elegantly handle the data size mismatch in multlingual translation is still a trendy topic. 

We also see that **scalarizaiton** and **gradient manipulation** modifies two different parts of the gradient: scalariation mostly operates on the magnitude  and projection mostly operates on the direction (but magnitude is also involved). 

#### Future Directions

I will throw some of my thoughts on future directions here: all of them are chanllenging but exciting to pursue and I envision myself working on these directions. 

- Find better ways to do dynamic scalarization without additional computational overhead - also needs to have strong improvements over Choi et al., 2023

- Understand the optimization landscape of multilingual translation - and find which findings generalizes to other deep multi-task settings (e.g. instruction tuning, LM pre-training) and also strong generalizability (e.g. zero-shot translation). Something Orhan Firat told me about is that methods that many-to-one mutllingual MT behaves more like general multi-task learning while it is hard to yield large imporvements on the one-to-many setting.

- Scale up! Langauge Model pre-training needs to find weights for each domain (PaLM 2) - this work did a grid search on the weights, which is expensive. There are many open questions: should these weights be static or dynamic? what is the most important factor when findings these weights [[Xie et al., 2023a](https://arxiv.org/abs/2302.03169), [Xie et al., 2023b](https://arxiv.org/abs/2305.10429)] - as simple as size? what about quality? how to find scalable methods to measure pre-training data quality ? 
