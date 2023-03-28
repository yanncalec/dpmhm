# Few-Shot & Transfer Learning

---
author:
- Han Wang[^1]
date: 2023-02-14
title:  Unsupervised Few-Shot Detection of Signal Anomalies
---

In the context of IA powered manufacturing applications, ranging from
quality control to process monitoring, we aim at developping efficient
methods of anomaly detection for acoustic or vibrational signals
operating with small amount of training data. This "small data" scenario
is certainly a challenge for the classic deep learning methods, yet
becomes an increasingly common situation in many industrial domains.

The detection of signal anomalies (DSA) amounts to extract from data the
information about the physical process of manufacturing, which is in
general too complex to be fully understood. Moreover, real data of
abnormal states are relatively scarce and often expensive to collect.
For these reasons we privilege a data-driven approach under the
framework of Few-Shot Learning (FSL). FSL is a methodology constructed
on top of classical learning methods, in the goal of improving their
generalization capacity in the senario of few learning samples.

The subject is also related to transfer learning. Actually, we are
aiming general application scenarios with multi-modal signals collected
on different machines or simulated from physical models. Data can have
arbitrary length, possibly be multivariated and weakly labelled. The
model trained in some context needs to be easily transfered to another
similar but distinct context.

# Few-Shot Learning {#sec:FSL}

FSL designates methods aims at learning from few available samples.
There exists a large literature on the subject and diverse approaches
have been proposed. In the setting of a classification problem, a formal
description of the procedure is given below. At the first stage a FSL
method is trained on some tasks. Then at the second stage a new task
$\tau$ containing $N$ classes unseen during the training stage is given
for validation. From $\tau$ a dataset $\mathcal{S}$ composing at most
$K$ samples per class ($N$-way and $K$-shot labelled data, typically $K$
is small) is drawn. Using $\mathcal{S}$ as support, the trained FSL
method yields a classifier $\psi_\mathcal{S}$ which returns the hidden
state of a sample. To evaluate the performance of $\psi_\mathcal{S}$ we
test it on a query dataset $\mathcal{Q}$ drawn independently also from
$\tau$.

A FSL method seen as a mapping $\mathcal{S}\mapsto \psi_\mathcal{S}$ can
be parameterized by one (or more) deep neural network (DNN) which is
learned during the training stage (where one is not limited by the
scarcity of data). The original sample space is transformed to some
feature space in which samples of different classes are optimaly
separated, and the optimality of the transformation is determined by the
choice of the loss function or the training criteria. The training
objective can be written in a general expression as $$\begin{aligned}
    \label{eq:FSL_training_obj}
    \min_{{\bm{\eta}}}
\ifthenelse{\equal{\tau\sim{\mathcal{G}}}{}}{\mathop{\mathrm{\mathbb{E}}}\left(
\ifthenelse{\equal{\mathcal{S}\sim\tau}{}}{\mathop{\mathrm{\mathbb{E}}}\left(
\ifthenelse{\equal{\mathcal{Q}\sim\tau}{}}{\mathop{\mathrm{\mathbb{E}}}\left( \sum_{(x,n)\in\mathcal{Q}}\mathcal{L}_{\bm{\eta}}(x, n; \mathcal{S}) \right)}{\mathop{\mathrm{\mathbb{E}}}_{\mathcal{Q}\sim\tau}\left( \sum_{(x,n)\in\mathcal{Q}}\mathcal{L}_{\bm{\eta}}(x, n; \mathcal{S}) \right)}%
 \right)}{\mathop{\mathrm{\mathbb{E}}}_{\mathcal{S}\sim\tau}\left(
\ifthenelse{\equal{\mathcal{Q}\sim\tau}{}}{\mathop{\mathrm{\mathbb{E}}}\left( \sum_{(x,n)\in\mathcal{Q}}\mathcal{L}_{\bm{\eta}}(x, n; \mathcal{S}) \right)}{\mathop{\mathrm{\mathbb{E}}}_{\mathcal{Q}\sim\tau}\left( \sum_{(x,n)\in\mathcal{Q}}\mathcal{L}_{\bm{\eta}}(x, n; \mathcal{S}) \right)}%
 \right)}%
 \right)}{\mathop{\mathrm{\mathbb{E}}}_{\tau\sim{\mathcal{G}}}\left(
\ifthenelse{\equal{\mathcal{S}\sim\tau}{}}{\mathop{\mathrm{\mathbb{E}}}\left(
\ifthenelse{\equal{\mathcal{Q}\sim\tau}{}}{\mathop{\mathrm{\mathbb{E}}}\left( \sum_{(x,n)\in\mathcal{Q}}\mathcal{L}_{\bm{\eta}}(x, n; \mathcal{S}) \right)}{\mathop{\mathrm{\mathbb{E}}}_{\mathcal{Q}\sim\tau}\left( \sum_{(x,n)\in\mathcal{Q}}\mathcal{L}_{\bm{\eta}}(x, n; \mathcal{S}) \right)}%
 \right)}{\mathop{\mathrm{\mathbb{E}}}_{\mathcal{S}\sim\tau}\left(
\ifthenelse{\equal{\mathcal{Q}\sim\tau}{}}{\mathop{\mathrm{\mathbb{E}}}\left( \sum_{(x,n)\in\mathcal{Q}}\mathcal{L}_{\bm{\eta}}(x, n; \mathcal{S}) \right)}{\mathop{\mathrm{\mathbb{E}}}_{\mathcal{Q}\sim\tau}\left( \sum_{(x,n)\in\mathcal{Q}}\mathcal{L}_{\bm{\eta}}(x, n; \mathcal{S}) \right)}%
 \right)}%
 \right)}%

    % \Exp[\tau\sim\Ptask]{\Exp[\lS\sim\tau]{\Exp[\lQ\sim\tau]{\sum_{(x,n)\in\lQ}\lL(\set{\phi_\veta(x;\lS_{n'})}_{n'}, n)}}}
\end{aligned}$$ where $\tau\sim{\mathcal{G}}$ means sampling a random
task from a class of tasks ${\mathcal{G}}$, and
$\mathcal{L}_{\bm{\eta}}$ is a loss function depending on the DNN
${\bm{\eta}}$. In particular this model includes the so-called "metric
learning" approach for FSL, in which the same condition of $N$-way and
$K$-shot of the validation stage is used for the training stage.[^2]

## Metric learning

A lot of training strategies for the objective
[\[eq:FSL_training_obj\]](#eq:FSL_training_obj){reference-type="eqref"
reference="eq:FSL_training_obj"} can be seen as a metric learning
problem, in which some metric/similarity $d_{\bm{\eta}}$ between
transformed samples is involved. In the following we write
$\mathcal{S}=\cup_n \mathcal{S}_n$ *i.e. *the partition of the support
dataset by class and define
$\mu_{\bm{\eta}}(\mathcal{S}_n):=\frac{1}{\left| \mathcal{S}_n \right|}\sum_{(x,n)\in\mathcal{S}_n} {\bm{\eta}}(x)$,
*i.e. *the centroid of transformed observations in $\mathcal{S}_n$. Then
the posterior distribution can be recovered as $$\begin{aligned}
    \label{eq:posterior_cat}
    \hat P(n|x) = \frac{e^{d_{\bm{\eta}}(x, \mathcal{S}_n)}}{\sum_{n'} e^{d_{\bm{\eta}}(x, \mathcal{S}_{n'})}}
    % \hat P(n|x) = \frac{e^{d(\eta(x), \mu_\veta(\lS_n))}}{\sum_{n'} e^{d(\eta(x), \mu_\veta(\lS_{n'}))}}
\end{aligned}$$ Then the classifier can be taken as
$\psi_\mathcal{S}(x) = \operatornamewithlimits{arg\,max}_n \hat P(n|x)$.

#### Prototypical Network [@snell_prototypical_2017]

This method uses the metric $$\begin{aligned}
    \label{eq:protonet_dist}
    d_{\bm{\eta}}(x,\mathcal{S}_n) = d({\bm{\eta}}(x), \mu_{\bm{\eta}}(\mathcal{S}_n))
\end{aligned}$$ with $d$ being a fixed Bregman distance *e.g. *the
squared euclidean distance or the cosine similarity, and the centroid
$\mu_{\bm{\eta}}(\mathcal{S}_n)$ is the prototype representing the class
$n$. The loss function is the cross-entropy between the binary
$\delta_n(\cdot)$ and the posterior $\hat P(\cdot|x)$: $$\begin{aligned}
    \label{eq:protonet_loss}
    \mathcal{L}_{\bm{\eta}}(x,n;\mathcal{S}) = \operatorname{D_\mathtt{KL}}\left( \delta_n(\cdot)||\hat P(\cdot|x) \right) =  -\log \hat P(n|x)
\end{aligned}$$

### Methods for one-shot learning

This is a special case of few-shot learning where only one sample per
class is given, *i.e. *$\mathcal{S}=\left\{ (x_n,n) \right\}_n$. The
metric in this setting takes the form $d_{\bm{\eta}}(x,x')$ but can be
adapted to FSL by passing *e.g. *the centroid
$\mu_{\bm{\eta}}(\mathcal{S}_n)$ as the second argument.

#### Siamese Network [@koch_siamese_2015]

This method transforms pairwisely a query and a support sample via the
same DNN ${\bm{\eta}}$ before comparing them. The similarity here is
$$\begin{aligned}
    \label{eq:siamese_dist}
    d_{\bm{\eta}}(x,x')=\sigma\left( \left\| {\bm{\eta}}(x)-{\bm{\eta}}(x') \right\|_1 \right)
\end{aligned}$$ with $\sigma$ being the sigmoid function. The binary
cross-entropy is used as the loss function $$\begin{aligned}
    \mathcal{L}_{\bm{\eta}}(x,n;\mathcal{S})
    = -\sum_{(x',n')\in\mathcal{S}} \delta_{n,n'} \log d(x,x') + \left( 1-\delta_{n,n'} \right)\log \left( 1-d(x,x') \right)
\end{aligned}$$ In practice the support dataset $\mathcal{S}$ and the
query dataset $\mathcal{Q}$ are sampled in such a way that the number
positive and negative pairs (a pair is positive if they are from a same
class) are balanced.

#### Matching Network [@vinyals_matching_2016]

Similar to Prototype Network, but the support set $\mathcal{S}$ is used
as a memory sequence to embed the query and support samples separately
via two distinct DNNs $f_\mathcal{S}$ and $g_\mathcal{S}$. The metric is
$$\begin{aligned}
    d_{\bm{\eta}}(x,x') = d(f_\mathcal{S}(x), g_\mathcal{S}(x'))
\end{aligned}$$ for some fixed $d$ *e.g. *the cosine similarity. These
DNN are learned together with ${\bm{\eta}}$ using the same loss function
[\[eq:protonet_loss\]](#eq:protonet_loss){reference-type="eqref"
reference="eq:protonet_loss"}.

#### Relation Network [@sung_learning_2018]

This method has a similare architecture to the Siamese Network but
rather than prescribe a metric it is parameterized by a DNN
${\bm{\phi}}$ $$\begin{aligned}
    \label{eq:relationnet_dist}
    d_{\bm{\eta}}(x,x') = {\bm{\phi}}({\bm{\eta}}(x), {\bm{\eta}}(x'))
\end{aligned}$$ which is learned together with ${\bm{\eta}}$ under a
non-standard loss function: $$\begin{aligned}
    \label{eq:relationnet_loss}
    \mathcal{L}_{\bm{\eta}}(x,n;\mathcal{S}) = \sum_{(x',n')\in\mathcal{S}} \left| d_{\bm{\eta}}(x, x')-\delta_{n,n'} \right|^2
\end{aligned}$$ As in the Siamese Network the sampling of $\mathcal{S}$
and $\mathcal{Q}$ has to maintain the balance between positive and
negative pairs.

## Other approaches

#### Transfer learning

It is believed that a central challenge to many learning problems is the
training of a good feature extractor. From there it can be coupled with
traditional classification methods (*e.g. *logistic regression, support
vector machine) to solve the problem of FSL. The efficiency of such a
"Deep + Shallow" approach has been reported recently by multiple authors
[@chen_new_2020; @sun_meta-transfer_2019; @muller_analysis_2020] where
some well-established architectures pretrained on large scale data for
general purpose task are used as ${\bm{\eta}}$ (by dropping the last
layer and adding eventually a fine-tuning step).

#### Meta learning

Another notable approach to FSL is the so-called meta learning problem,
where a meta learner is trained to quickly adapt a base learner to the
new task $\tau$ by doing a gradient descent step, see
[@finn_model-agnostic_2017] and the related works
[@ravi_optimization_2017].
<!--
::: thebibliography
10

Y. Chen, X. Wang, Z. Liu, H. Xu, and T. Darrell. A New Meta-Baseline for
Few-Shot Learning. , Apr. 2020. arXiv: 2003.04390.

C. Finn, P. Abbeel, and S. Levine. Model-agnostic meta-learning for fast
adaptation of deep networks. In *International Conference on Machine
Learning*, pages 1126--1135. PMLR, 2017.

Y. Kawaguchi, K. Imoto, Y. Koizumi, N. Harada, D. Niizumi, K. Dohi,
R. Tanabe, H. Purohit, and T. Endo. Description and Discussion on DCASE
2021 Challenge Task 2: Unsupervised Anomalous Sound Detection for
Machine Condition Monitoring under Domain Shifted Conditions. , June
2021. arXiv: 2106.04492.

G. Koch, R. Zemel, and R. Salakhutdinov. Siamese neural networks for
one-shot image recognition. In *ICML deep learning workshop*, volume 2.
Lille, 2015.

Y. Koizumi, Y. Kawaguchi, K. Imoto, T. Nakamura, Y. Nikaido, R. Tanabe,
H. Purohit, K. Suefusa, T. Endo, and M. Yasuda. Description and
discussion on DCASE2020 challenge task2: Unsupervised anomalous sound
detection for machine condition monitoring. , 2020.

Y. Koizumi, S. Murata, N. Harada, S. Saito, and H. Uematsu. : Few-shot
learning for anomaly detection to minimize false-negative rate with
ensured true-positive rate. In *ICASSP 2019-2019 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pages
915--919. IEEE, 2019.

Y. Koizumi, S. Saito, H. Uematsu, Y. Kawachi, and N. Harada.
Unsupervised Detection of Anomalous Sound Based on Deep Learning and the
Neyman--Pearson Lemma. , 27(1):212--224, Jan. 2019. Conference Name:
IEEE/ACM Transactions on Audio, Speech, and Language Processing.

C. Li, S. Li, A. Zhang, Q. He, Z. Liao, and J. Hu. Meta-learning for
few-shot bearing fault diagnosis under complex working conditions. ,
439:197--211, 2021. Publisher: Elsevier.

P. Malhotra, A. Ramakrishnan, G. Anand, L. Vig, P. Agarwal, and
G. Shroff. -based Encoder-Decoder for Multi-sensor Anomaly Detection. ,
July 2016. arXiv: 1607.00148.

R. Müller, S. Illium, F. Ritz, and K. Schmid. Analysis of Feature
Representations for Anomalous Sound Detection. , 2020.

S. Ravi and H. Larochelle. Optimization as a Model for Few-Shot
Learning. In *5th International Conference on Learning Representations,
ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track
Proceedings*. OpenReview.net, 2017.

L. Ruff, J. R. Kauffmann, R. A. Vandermeulen, G. Montavon, W. Samek,
M. Kloft, T. G. Dietterich, and K.-R. Müller. A unifying review of deep
and shallow anomaly detection. , 2021. Publisher: IEEE.

J. Snell, K. Swersky, and R. Zemel. Prototypical networks for few-shot
learning. In *Proceedings of the 31st International Conference on Neural
Information Processing Systems*, NIPS'17, pages 4080--4090, Red Hook,
NY, USA, Dec. 2017. Curran Associates Inc.

Q. Sun, Y. Liu, T.-S. Chua, and B. Schiele. Meta-transfer learning for
few-shot learning. In *Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition*, pages 403--412, 2019.

F. Sung, Y. Yang, L. Zhang, T. Xiang, P. H. Torr, and T. M. Hospedales.
Learning to compare: Relation network for few-shot learning. In
*Proceedings of the IEEE conference on computer vision and pattern
recognition*, pages 1199--1208, 2018.

O. Vinyals, C. Blundell, T. Lillicrap, K. Kavukcuoglu, and D. Wierstra.
Matching networks for one shot learning. In *Proceedings of the 30th
International Conference on Neural Information Processing Systems*,
NIPS'16, pages 3637--3645, Red Hook, NY, USA, Dec. 2016. Curran
Associates Inc.
::: -->

[^1]: Email: han.wang@cea.fr

[^2]: Note that this is only a way of presenting the training data but
    not imposing the data scarcity as in the validation stage,
    *i.e. *the number of episodes is not limited.
