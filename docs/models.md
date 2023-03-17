# Models

Models implemented in `dpmhm`:

- Supervised learning: VGGish networks
- Unsupervised learning: Auto-encoder and VAE
- Self-supervised learning: contrastive and generative models
- Semi-supervised learning: hybrid models targeting transfer learning.

<!-- | Model | Type | Input |Reference |
| :---- | :--- | :---- |:-------- |
| VGGish | SL |  | -->

<!-- ### Overview of architectures
CNN, AE...
 -->

## Supervised learning
For labelled datasets only, representations can be learned by solving the classification task. The following models are implemented:

- VGGish[@simonyan_very_2014,@hershey_cnn_2017]
- VGGish for signal (WIP)


## Unsupervised learning

UL can be applied on datasets without label about the faulty state. The following models are implemented:

- Auto-encoder [@vincent_stacked_2010, @qian_new_2021]
- Monte-Carlo EM
- Variational Auto-encoder (VAE) [@kingma_auto-encoding_2013, @an_variational_2015]

<!-- - Clustering based unsupervised learning, for weakly labelled datasets: e.g. only information on the operating condition or machine type is available. -->

<!-- ### Generative learning -->

## Self-supervised learning

SSL models can be divided into contrastive and generative ones. The following models are implemented:

- SimCLR[@chen_simple_2020]

<!-- ```python
from dpmhm.models.ssl import simclr
``` -->


## Semi-supervised learning
