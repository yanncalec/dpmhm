# Models

Multiple baselines are implemented in `dpmhm` can be used for representation learning, including

- Supervised learning: VGGish networks
- Unsupervised learning: Auto-encoder and VAE
- Self-supervised learning: contrastive and generative models

See the description below for the supported baseline models of each category.

<!-- | Model | Type | Input |Reference |
| :---- | :--- | :---- |:-------- |
| VGGish | SL |  | -->

### Overview of architectures
CNN, AE...

## Supervised learning
For labelled datasets only, representations can be learned by solving the classification task. The following models are implemented:

- VGGish[@simonyan_very_2014,@hershey_cnn_2017]
- VGGish for signal (WIP)

<!-- ### VGGish networks
TODO:

- explain VGGish NN and highlight some results of transfer learning on CWRU and other datasets.
- implementation details of VGGish for signal.
 -->

## Unsupervised learning
UL can be applied on datasets without label about the faulty state. The following models are implemented:

- Auto-encoder [@vincent_stacked_2010, @qian_new_2021]
- Variational Auto-encoder (VAE) [@kingma_auto-encoding_2013, @an_variational_2015] (WIP)
<!-- - Clustering based unsupervised learning, for weakly labelled datasets: e.g. only information on the operating condition or machine type is available. -->

### Domain invariance
Domain level information (e.g. operating conditions or dataset ID) can be exploited (even for dataset without label) together with AE for domain invariant representation learning. [@li_perspective_2022]

<!-- $$
\loss_{\small{AE}} + \loss_{\small{CE}}
$$ -->


## Self-supervised learning
SSL models can be divided into contrastive and generative ones. The following models are implemented:

- SimCLR[@chen_simple_2020]

<!-- ```python
from dpmhm.models.ssl import simclr
``` -->

<!-- ### Generative learning -->
