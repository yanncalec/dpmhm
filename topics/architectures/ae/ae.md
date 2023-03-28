# Unsupervised learning using Auto-Encoder
Unsupervised learning refers to techniques without explicit supervision or labels, and the goal is to identify patterns, structure, and relationships in the data without prior knowledge of what the patterns or relationships are. Popular approaches of unsupervised learning include:

- Clustering: group data points together based on some similarity metric. Examples: k-means, hierarchical clustering, and DBSCAN etc.
- Dimensionality Reduction: reduce the dimensionality of datasets while retaining the important information. Examples: Principal Component Analysis (PCA), t-SNE, and autoencoders.
- Generative Models: learn the underlying distribution of the data for tasks such as data generation and anomaly detection. Examples: Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and Normalizing Flows.

Hereafter we briefly review the Auto-Encoder and its probabilistic version.

## Auto-Encoder and its variants
The autoencoder architecture consists of an encoder $f_\theta$ such that $z=f_\theta(x)$ is a low-dimensional representation of the input $x$,
and a decoder $g_\phi$ such that $\hat x = g_\phi(z)$ is an approximation of $x$ from the latent representation $z$. Here $\theta, \phi$ are the parameters of the two neural networks that can be learned by minimizing the approximation error:

\begin{align}
\label{eq:ae}
\min_{\theta, \phi} \norm{x- g_\phi \circ f_\theta(x)}^2
\end{align}

By minimizing this error, the network $f_\theta$ learns to encode the input data into a lower-dimensional space while preserving the most important information needed for reconstruction, and the network $g_\phi$ learns to decode the representation back to the observation so that it is close to the original one. This can be useful for a variety of tasks, such as dimensionality reduction, feature extraction, and data generation. It can be also seen as a type of self-supervised learning, where the network learns to generate a target output based on an input without explicit supervision.

<!-- #### Architecture -->
Choosing multiple convolutional layers as architectures for the encoder and decoder we obtain the so called convolutional AE stacks (CAES).

### AEs for anomaly detection
Anomalies or outliers can be identified by computing the reconstruction error of the AE: If the reconstruction error is higher than a certain threshold, then the input is classified as an anomaly. For this, we train the AE on a dataset that only contains normal data. The AE will learn the distribution of the normal data in a compressed space. Anomalies are then detected by computing the reconstruction error of new data points and comparing them to a threshold learned from the training data.

### Sparse AE
Regularization terms can be introduced in the loss function $\eqref{eq:ae}$ above. Adding a L1 regularization (or KL divergence term) gives the sparse autoencoder which enforces a sparsity constraint on the number of parameters (neurons) that activate during training. This encourages the autoencoder to learn a compressed representation of the input data, which can be useful for tasks such as feature extraction and dimensionality reduction.
<!-- \begin{align}
\label{eq:sparse_ae}
\min_{\theta, \phi} \norm{x- g_\phi \circ f_\theta(x)}^2 + \lambda_\theta \norm{\theta}_1 + \lambda_\phi \norm{\phi}_1
\end{align} -->

### Denoising AE
A denoising autoencoder is a variant that allows to learn robust representations. During training, the network is presented with a clean target and a corrupted version, such as adding Gaussian noise or randomly masking parts of data, and the loss is computed based on the difference between the reconstruction and the clean target.

<!-- ## Probabilistic graphical models with latent variables -->

## Variation Auto-Encoder
Variation Auto-Encoder (VAE) is a kind of the probabilistic graphical models with latent variables. As above, let $x$ be a data point and $z$ be some latent variable. In the context of MHM, we can choose these variables to be

- $x$: patch of spectrogram, eventually with the information of frequency range
- $z$: mixture of class label and generatve model

Denote by $p_\theta, q_\phi$ the p.d.f. of the distribution of observation and latent state, respectively. The true posterior $p_\theta(z|x)$ is most often intractable and the idea is to use the variational posterior $q_\phi(z|x)$ as approximation. The principle of VAE is to maximize a lower bound of the log-evidence $\log p_\theta(x)$ termed ELBO, which is defined as:

\begin{align}
\label{eq:ELBO}
\ELBO &:= \Exp_{z\sim q_\phi(\cdot|x)}[\log p_\theta(x|z)] - \KL{q_\phi(z|x) }{p_\theta(z)}\\
&= \Exp_{z\sim q_\phi(\cdot|x)} [\log p_\theta(x,z) - \log q_\phi(z|x)]
\end{align}

here, $p_\theta(x|z)$ is interpreted as a probabilistic decoder and $q_\phi(z|x)$ as an encoder (or a recognition model). This is indeed a lower bound of the log-evidence because it holds

$$
\log p_\theta(x) = \ELBO + \KL{q_\phi(\cdot|x)}{p_\theta(\cdot|x)} \geq \ELBO.
$$

<!-- The most useful form of ELBO is however $\eqref{eq:ELBO}$.  -->
As in the case of classical AE, $\theta, \phi$ are the parameters of NNs and are learned by maximizing ELBO using SGD. Either forms of ELBO can be used, and the derivative of $\phi$ can be evaluated by Monte Carlo using a reparameterization trick, e.g.:

\begin{align}
z = g_\phi(\epsilon; x) = \mu_\phi(x) + \sigma_\phi(x)\epsilon
\end{align}

with $\epsilon \sim \Normal{0}{1}$ and $\mu_\phi, \sigma_\phi$ NNs mapping $x$ to the parameters of the variational posterior.

<!-- While the KL term in ELBO can be made analytic -->

<!-- $$
\max_{\theta, \phi} \Exp_{\epsilon \sim p_\epsilon}[\log p_\theta(x|g_\phi(\epsilon))] - \KL{q_\phi(g_\phi(\epsilon)|x) }{p_\theta(g_\phi(\epsilon))}
$$ -->

#### Application in anomaly detection
VAEs have been successfully used for anomaly detection in several applications. In the same spirit as the classical AE, we train first the VAEs $q_\phi(z|x), p_\theta(x|z)$ on normal data, then given a new data point we evaluate the log-evidence

\begin{align}
\label{eq:log_evidence}
\log p_\theta(x) = \log \Exp_{z\sim q_\phi(\cdot|x)} \bracket{\frac{p_\theta(x,z)}{q_\phi(z|x)}}
\end{align}

using Monte Carlo. The decision can be made by comparing the log-evidence to a threshold learned on the training data. In place of $\eqref{eq:log_evidence}$, the ELBO or the reconstruction probability (first term in ELBO) can be used:

$$
\Exp_{z\sim q_\phi(\cdot|x)} \bracket{\log p_\theta(x|z)}
$$

<!-- There are also several variations and extensions of VAEs that have been proposed for anomaly detection, such as using adversarial training or incorporating auxiliary information into the VAE architecture. -->


### Monte-Carlo EM

$$
p_\theta(x) = \Exp_{z\sim p_\theta(\cdot|x)}[p_\theta(x|z)^{-1}]^{-1}
$$

