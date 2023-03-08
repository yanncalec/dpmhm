# Machine Health Monitoring & Deep Learning

We briefly describe the advances in MHM, in particular the end-to-end intelligent fault diagnosis, under the lens of deep learning. More details can be found in recent reviews & monographs [@lei_applications_2020, @zhang_deep_2020, @rezaeianjouybari_deep_2020]. A classical workflow in MHM [@randall_rolling_2011, @qin_survey_2012, @lei_machinery_2018] is typically composed of the following steps:

- data acquisition,
- feature extraction,
- model construction,
- decision making.

## Data acquisition
A major body of work in MHM is based on vibration monitoring of components. This technique involves the analysis of vibration data coming from vital components of a rotating machine to detect features which reflect the operational state of machinery. The analysis leads to the identification of potential failures and their causes, and makes it possible to perform efficient preventive maintenance. The same signal analysis techniques can also be applied to signals of alternative modalities, such as acoustic emission, current-voltage, rotating speed etc, which may provide useful complementary information in certain applications [@mey_condition_2021, @pandya_fault_2013]. Datasets provided by `dpmhm` contain essentially vibration signals.

Recent studies[@smith_rolling_2015, @neupane_bearing_2020] reveal some important aspects that the data acquisition in view of fault diagnostic should follow. For example, as stated in [@smith_rolling_2015]:
<!-- !!! quote -->
> Faults in bearings often manifest themselves at high frequencies, so the use of a high sampling rate – perhaps greater than 40 kHz – is recommended.

Some situations are frequently encountered in the data acquisition stage, e.g.
sample imbalance (dominant healthy samples), domain imbalance (dominant samples from the source domain), inaccuracy in labelling and segmentation etc.

## Feature extraction
In a classical workflow, feature extraction (or feature engineering) consists in transforming the raw signals (vibration/acoustic etc) into advanced forms using signal processing tools [@rai_review_2016]. Premilinary signal processings, e.g. signal separation and denoising, may be applied at this stage. Some popular features widely adopted in MHM are:

- Linear Predictive Coding (LPC)
- Kurtosis
- Empirical Mode Decomposition (EMD)
- [Cyclostationarity](https://en.wikipedia.org/wiki/Cyclostationary_process),
- Wiener-Ville Distribution
- Time-Scale Analysis: Wavelet Packet Transform (WPT)
- Time-Frequency Analysis: spectrogram, cepstrum, [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), etc.

In place of raw signals, the handcrated features can be taken as input to a representation learning network. See discussions of representation learning below.

## Model construction
Over the last decades a lot of statistical machine learning models have been successfully applied to MHM, including support vector machine (SVM), k-nearest neighbors (KNN), random forest etc [@khan_review_2018, @liu_artificial_2018]. Compared to more recent end-to-end models based on deep learning, these "shallow" methods generally achieve lower accuracy in MHM tasks but enjoy better explainability. Some of them are implemented as baseline methods in the `dpmhm` package.

In the following we outline mainly the deep learning based models. Our view on tasks of MHM can be summarized as:

==Representation learning + Knowledge tranfer to downstream task==

In the first stage, a NN for representation is trained by solving tasks adapted to the training dataset. In the second stage, the learned representation NN is transfered to some downstream tasks on the test dataset (e.g. by taking the penultimate layer of the trained classifier and adding supplementary layers then fine-tuning it). Such a paradigm is general enough to cover a wide range of application, from intelligent fault diagnostic to RUL prediction.

### Architectures
Compared to classical "shallow" models, modern models based on deep learning can achieve higher accuracy while being more flexible and versatile, thus better suited for end-to-end system design. Depending on the specification of the dataset (e.g. unlabeled, unbalanced, limited) and the MHM task (diagnostic or prognostic [@rezaeianjouybari_deep_2020,]), different network architectures have been proposed and some most popular ones are [@zhang_deep_2020, @zhao_deep_2019]:

- Convolutional Neural Network (CNN) [@jiao_comprehensive_2020]
- Auto-Encoder (AE) [@yang_autoencoder-based_2022]
- Deep Belief Network (DBN) [@mohamed_acoustic_2012]
- Recurrent Neural Network (RNN)
- Generative Adversarial Network (GAN)
<!-- - Capsule -->

### Transfer learning
Transfer Learning (TL) is a machine learning methodology that focuses on applying knowledge gained while solving one problem to a different but related problem. It has received considerable attention in MHM and helps answer the question of whether a model trained on one type of machine/operating condition/fault can be adapted to another type [@li_systematic_2020, @zheng_cross-domain_2019,], which is central to the application in realistic scenarios.

#### Few-shot learning
Moreover, the real-case mechanical data has obviously unlabeled and unbalanced characteristics [@zhang_intelligent_2022]. Actually:

- Compared to image datasets like ImageNet, vibration/acoustic/electrical signals are inherently more difficult to label, and manual labels can be inaccurate or incomplete.
- Damage to rotating machinery is often irreversible. Unless initiated intentionally (which can be costly), anomalous data is collected less frequently than normal data. Moreover, different types of faults can also occur at different rates.

Few-shot learning (FSL) aims to improve the sample efficiency of learning, or in other words, to learn fast. This may be particularly relevant for MHM due to unbalanced data characteristics [@zhang_limited_2019, @li_meta-learning_2021]. For example, diagnosis of a certain type of fault must be performed with a small amount of labeled data. FSL can be viewed as a specific form of TL in that intelligence gained on a generic context should be adaptable to a new context using only a small amount of information from the new context.

### Representation learning
<!-- Given the important role of tranfer learning as well as the success of invariant feature learning by various deep architectures, it seems natural to separate a MHM task into the up-stream task of representation learning and the down-stream task of transfer learning. -->

<!-- #### Self-supervised learning -->

Self-supervised learning (SSL) refers to a machine learning paradigm
to discover general representations from large-scale data without requiring human annotations, which is an expensive and time-consuming task.
Such representations are useful for downstream learning tasks. Our view is that self-supervised representation learning is the key to transfer learning and few-shot learning in the context of MHM.

Classical representations of signals as time-frequency or time-scale features are manually designed and not adaptive to the signal class or to the ultimate task (e.g. diagnosis in MHM). However, in self-supervised learning, the representation is learned end-to-end directly from signals (or a classical form of representation) without the label information by solving some pre-tasks, which are generally discriminative or generative.

<!-- Among the models implemented in `dpmhm`, e.g. supervised (CNN) or unsupervised (auto-encoder), the SSL models take center stage: representations are first obtained by SSL on a meta-dataset (e.g. union of multiple datasets) then tranfered to a new dataset via a specific downstream task (e.g. diagnosis or prognosis). -->

## Decision making
A final step in the MHM workflow is to making decision with the outcomes of previous steps. This invovles often determining the health state of a machine given a test signal, which can be either diagnostic (a classification problem, i.e. to decide the type of fault) or pronostic (a prediction problem, i.e. to decide the RUL).

<!-- ### Intelligent fault diagnostics -->

<!-- ### RUL & prognostics

Prognostics aims at determining whether a failure of an engineered system (e.g., a nuclear power plant) is impending and estimating the remaining useful life (RUL) before the failure occurs. The traditional data-driven prognostic approach is to construct multiple candidate algorithms using a training data set, evaluate their respective performance using a testing data set, and select the one with the best performance while discarding all the others.

This approach has three shortcomings: (i) the selected standalone algorithm may not be robust; (ii) it wastes the resources for constructing the algorithms that are discarded; (iii) it requires the testing data in addition to the training data.

To overcome these drawbacks, this paper proposes an ensemble data-driven prognostic approach which combines multiple member algorithms with a weighted-sum formulation. Three weighting schemes, namely the accuracy-based weighting, diversity-based weighting and optimization-based weighting, are proposed to determine the weights of member algorithms. The k-fold cross validation (CV) is employed to estimate the prediction error required by the weighting schemes. The results obtained from three case studies suggest that the ensemble approach with any weighting scheme gives more accurate RUL predictions compared to any sole algorithm when member algorithms producing diverse RUL predictions have comparable prediction accuracy and that the optimization-based weighting scheme gives the best overall performance among the three weighting schemes. -->