---
layout: post
title:  "Reward is enough context : Improving Zero-Shot Generalization in Reinforcement Learning"
date:   2024-05-03 11:20:25 +0100
categories: reinforcement-learning
lang: en
ref: unsupervised-learning
use_math: true
---


{% include _mathjax_support.html%}

# Introduction

Reinforcement learning (RL) has achieved remarkable success in a variety of domains, from playing games to robotic control. However, one of the key challenges in RL is generalizing to unseen tasks or environments, known as zero-shot generalization. Zero-shot generalization refers to the ability of an RL agent to perform well on tasks it has never seen before, without any additional training or fine-tuning. This capability is crucial for real-world applications where agents must adapt to new scenarios without extensive retraining.

basics of rl : agent, action, environment, rewards

|![rl_schema](/assets/images/reward_context/rl_schema.png)|
|:--:| 
| Simple Reinforcement Learning schema |

example of the Ant environment in Brax : Soft actor critic learns a convincing policy in a few thousands of training steps, but fails when evaluated in an altered environment, e.g when the mass of the robot is changed.
|![ant_ood](/assets/images/reward_context/ant_ood.png)|



How to learn a policy that is robust to changes in dynamics ? 

# Learning a robust policy

contextual reinforcement learning
|![crl_cartpole](/assets/images/reward_context/cartpole_contexts.png)|
|:--:| 
| Different lenghts for cartpole |

## Domain randomization

One approach to learning a robust policy is to train the agent in a diverse set of environments, hoping that it will learn a policy that generalizes well to unseen tasks. This approach is known as domain randomization, where the agent is trained on a distribution of environments with varying dynamics, such as mass, friction, or gravity. The idea is that by exposing the agent to a wide range of environments during training, it will learn a policy that is robust to changes in dynamics and can generalize to unseen tasks.

However, domain randomization is limited by the fact that we still learn a single policy that must work across all environments. This can lead to suboptimal performance in some environments, as the agent must compromise to perform well across the entire distribution.
Lets take for example the [Gymnamsium] MountainCar environment, where an RL agent must learn to drive a car up a hill by applying force to the car. 

|![crl_cartpole](/assets/images/reward_context/mountaincar.gif)|
|:--:| 
| mountaincar task |

In this environment, the power parameter controls the amplitude of the force applied to the car. When the agent is trained with domain randomization on a range of power values, it leads to a failing policy.



contextual reinforcement learning
|![crl_cartpole](/assets/images/reward_context/train_curve_dom_rand.png)|
|:--:| 
| reward curve during training |


## Providing the context directly to the agent : Explicit identification

Another approach to learning a robust policy is to provide the agent with explicit information about the task or environment it is currently facing. This approach, known as explicit identification, involves giving the agent additional context or metadata about the task, by concatenating it to the state. By providing the agent with this information, it can use it to adapt its policy to the current task and improve its performance.

In the previous example of the MountainCar environment, concatenating the power parameter to the state allows the agent to learn a policy that is robust to changes in power values during training.

However, explicit identification requires the agent to have access to the context or metadata during training and evaluation, which may not always be available in real-world scenarios. 

## Inferring the context from environment interactions

An alternative approach to learning a robust policy is to infer the context or task information from the agent's interactions with the environment. This approach, known as implicit identification, involves training the agent to learn a representation of the context from the environment observations and rewards. By learning to infer the context from its interactions, the agent can adapt its policy to the current task without explicit context information.

This approach has been explored by [Evans et al., 2021](https://arxiv.org/abs/2203.05549). In their work, they propose to encode a the environment into a low-dimensional latent space, which is then used to condition the policy. 

After having collected a dataset of interactions from various contexts, a **context encoder** is first trained on a next-step prediction task. The context encoder is a neural network that takes the current state and action as input, as well as a list of (state, action, next state) interactions from the same context, and predicts the next observation. 

This task encourages the context encoder to learn a low-dimensional representation of the context that captures the relevant information for predicting the next observation. In our example of the MountainCar environment, the embedding space is highly correlated with the power parameter.

After this first trainin step, we train the policy algorithm by appending the latent representation of the context encoder to the state at each timestep. This allows the policy to adapt to the current context and improve its performance during training.

However, this approach requires a large amount of interaction data to train the context encoder effectively, which can be challenging to collect in practice. In addition, the training is divided into two separate stages with different objectives, which might lead to learned representations that are not optimal for the policy task. We propose to join those two steps into a single training process, where the context encoder is trained jointly with the policy algorithm.



## Learning context from rewards

In our approach, we propose to learn the context directly from the rewards received by the agent during training. The key idea is that the rewards provide implicit information about the task or environment the agent is currently facing, which can be used to infer the context. The context encoder is trained by backpropagating the loss of the policy algorithm through the context encoder.

In the MountainCar example, this training method leads to an higher performance of the agent during training than the predicive model of [Evans et al., 2021](https://arxiv.org/abs/2203.05549). How well does this approach generalize to unseen context values ?


# Can behavior-specific context embeddings improve generalization ?




Recent advances in machine learning have led to deep neural networks being commonly applied to electroencephalogram (EEG) data for a variety of decoding tasks. EEG is a non-invasive method that records the electrical activity of the brain using electrodes placed on the scalp. While deep learning models can achieve state-of-the-art performance on specialized EEG tasks, most EEG analyses focus on training task-specific models for one type of classification or regression problem ([Heilmeyer et al., 2018](https://www.researchgate.net/publication/330475732_A_Large-Scale_Evaluation_Framework_for_EEG_Deep_Learning_Architectures))

However, medical EEG recordings are often annotated with additional unstructured data in the form of free text reports written by neurologists and medical experts that can be exploited as a source of supervision. 

|![tuh_dataset](/assets/images/eegclip/tuh_dataset.jpg)|
|:--:| 
| Overview of an annotated EEG record |

In the computer vision domain, Contrastive Language–Image Pre-training ([Radford et al., 2021](https://arxiv.org/abs/2103.00020)) leverages this text-image pairing to learn visual representations that effectively transfer across tasks. Inspired by CLIP, we propose EEG-CLIP - a contrastive learning approach to align EEG time series data with corresponding clinical text descriptions in a shared embedding space. This work explores two central questions: (i) how textual reports can be incorporated into an EEG training pipeline, and (ii) to what extent this multimodal approach contributes to more general EEG representation learning.

We demonstrate EEG-CLIP's potential for versatile few-shot and zero-shot EEG decoding across multiple tasks and datasets. EEG-CLIP achieves nontrivial zero-shot classification results. Our few-shot results show gains over previous transfer learning techniques and task-specific models in low-data regimes. This presents a promising approach to enable easier analyses of diverse decoding questions through zero-shot decoding or training task-specific models from fewer training examples, potentially facilitating EEG analysis in medical research.

# Methodology

Contrastive self-supervised learning has recently emerged as a powerful approach for learning general visual representations. Models like CLIP are trained to align image $x_i$ and corresponding text $y_i$ embeddings by minimizing a contrastive loss $\mathcal{L}$:

$\mathcal{L}=\sum_{i=1}^{N}-\log \frac{\exp(\text{sim}(x_i,y_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(x_i,y_j)/\tau)}$

where sim(.) is a measure of similarity, such as cosine similarity, and $\tau$ is a temperature parameter that controls the softness of the distribution. This objective brings matching image-text pairs closer and separates mismatched pairs in the learned embedding space.

CLIP is trained on a vast dataset of 400 million image-text pairs from diverse Internet sources with unstructured annotations. Through this natural language supervision, CLIP develops versatile image representations that achieve strong zero-shot inference on downstream tasks by querying the aligned embedding space.

|![clip](/assets/images/eegclip/clip_illustration.png)|
|:--:| 
| Illustration of the constrastive learning process |

Inspired by the success of CLIP, we design EEG-CLIP as a contrastive learning framework for aligning EEG time series data with corresponding clinical text descriptions. EEG-CLIP consists of two encoder networks: an EEG encoder $f_{\theta}$ and a text encoder $g_{\phi}$. The EEG encoder $f_{\theta}$ maps the input EEG time series $x_i$ to a fixed-dimensional embedding vector $f_{\theta}(x_i)$, while the text encoder $g_{\phi}$ maps the corresponding clinical text description $y_i$ to an embedding vector $g_{\phi}(y_i)$.

The encoders are trained to minimize the contrastive loss $\mathcal{L}$, which encourages the embeddings of matching EEG-text pairs to be similar while pushing apart the embeddings of mismatched pairs. The similarity measure sim(.) used in EEG-CLIP is the cosine similarity between the normalized embeddings:

$\text{sim}(x_i, y_j) = \frac{f_{\theta}(x_i)^\top g_{\phi}(y_j)}{\lVert f_{\theta}(x_i)\rVert \lVert g_{\phi}(y_j)\rVert}$


# Experimental Setup
In this section, we describe the dataset used for training and evaluating EEG-CLIP, as well as the experimental settings and evaluation metrics employed to assess its performance. We focus on the Temple University Hospital EEG Corpus [Obeid and Picone, 2016](https://isip.piconepress.com/projects/tuh_eeg/html/overview.shtml), a large-scale dataset containing EEG recordings and their corresponding medical reports. The dataset's size and diversity make it well-suited for training deep learning models to learn general EEG representations.

## Dataset Description

The Temple University Hospital EEG Corpus [Obeid and Picone, 2016](https://isip.piconepress.com/projects/tuh_eeg/html/overview.shtml) is a comprehensive dataset containing over 25,000 EEG recordings collected from more than 14,000 patients between 2002 and 2015. The dataset's extensive size and variety of EEG recordings make it a valuable resource for training deep learning models to decode information such as pathology, age, and gender from EEG signals and generalize to unseen recordings.

For our experiments, we utilize the TUH Abnormal dataset (TUAB), a demographically balanced subset of the corpus with binary labels indicating the pathological or nonpathological diagnosis of each recording. The TUAB dataset is partitioned into a training set, consisting of 1,387 normal and 1,398 abnormal files, and an evaluation set, containing 150 normal and 130 abnormal files. The dataset encompasses a wide range of pathological conditions, ensuring a diverse representation of EEG abnormalities.

In addition to the binary pathology labels, each recording in the TUAB dataset is accompanied by several additional labels:

- "age": An integer representing the patient's age at the time of the recording.
- "gender": A string indicating the patient's gender, either "M" for male or "F" for female.
- "report": A medical report written in natural language, providing a detailed description of the EEG findings and clinical interpretation.

The medical reports in the TUAB dataset are structured into 15 distinct sections, each focusing on a specific aspect of the EEG analysis. the following provides an overview of these sections and their respective contents.

| Record Section                | Non-empty Entries | Average Word Count (Non-empty Entries) |
|--------------------------------|-------------------|----------------------------------------|
| IMPRESSION                     | 2971              | 16                                     |
| DESCRIPTION OF THE RECORD      | 2964              | 70                                     |
| CLINICAL HISTORY               | 2947              | 26                                     |
| MEDICATIONS                    | 2893              | 4                                      |
| INTRODUCTION                   | 2840              | 31                                     |
| CLINICAL CORRELATION           | 2698              | 31                                     |
| HEART RATE                     | 1458              | 2                                      |
| FINDINGS                       | 887               | 16                                     |
| REASON FOR STUDY               | 713               | 2                                      |
| TECHNICAL DIFFICULTIES         | 684               | 3                                      |
| EVENTS                         | 569               | 8                                      |
| CONDITION OF THE RECORDING     | 116               | 30                                     |
| PAST MEDICAL HISTORY           | 19                | 8                                      |
| TYPE OF STUDY                  | 16                | 3                                      |
| ACTIVATION PROCEDURES          | 9                 | 3                                      |


## Evaluation metrics 

Unlike models trained for a specific downstream task, EEG-CLIP aims to learn broadly useful representations that capture semantic relationships between EEG signals and text. As such, evaluation methods must aim to quantify the general quality and transferability of the learned representations.

Using the labels, meta-information, and medical reports provided in the TUAB dataset, we select four decoding tasks:

- "pathological": Decode whether the recording was diagnosed as normal or pathological.
- "age": Decode whether the age of the patient is smaller than or equal to 50, or greater than 50.
- "gender": Decode the declared gender of the patient.
- "medication": Decode whether the medical report contains at least one of the three most common anticonvulsant medications ("keppra", "dilantin", "depakote").


We design multiple methods to evaluate the model, as described in the following subsections.

### Classification 
In this method, we use the representations learned by the EEG encoder of EEG-CLIP as features for the four classification tasks. The encoder representations are kept frozen, while a classifier is trained and evaluated on the "train" and "eval" sections of TUAB. This enables us to assess whether EEG-CLIP has learned to compress the discriminative information relevant for the classification tasks into its embeddings.

We compare this approach against two baseline models with the same architecture as the EEG encoder:


- The first baseline is a fully trainable model. This provides an upper bound on performance since the model can directly optimize for each task. Comparing EEG-CLIP to the trainable model reveals how much room there is for improvement over the fixed EEG-CLIP features.
- The second baseline is a model trained on an unrelated task, whose features are frozen while only the classification head is trainable. As these task-unrelated representations are not specialized for the actual decoding tasks, this serves as a lower bound on expected performance. The gap between the lower baseline and EEG-CLIP quantifies the benefits of our contrastive approach that can learn from all information contained in the medical reports.


By situating EEG-CLIP between these upper and lower bounds, we can better isolate the contributions of the learned representations themselves. Smaller gaps to the trainable model and larger gaps from the task-unrelated features indicate higher-quality multimodal representations.

### Zero-shot classification
We also perform zero-shot evaluation, using the embeddings of class-specific text prompts as class prototypes for the trained EEG-CLIP model. For a given classification task, we define a typical prompt sentence for each class and calculate the distance of an EEG recording to those sentences in the shared embedding space. This allows us to measure the classification performance of EEG-CLIP without any training on the classification task labels.

|![zc](/assets/images/eegclip/zero_shot.png)|
|:--:| 
| Illustration of the zero-shot classification task |

| Task name   | Prompt a                          | Prompt b                          |
|-------------|-----------------------------------|-----------------------------------|
| pathological| "This is a normal recording"      | "This is an abnormal recording"   |
| age         | "The patient is over 50 years old"| "The patient is under 50 years old" |
| gender      | "The patient is male"             | "The patient is female"           |
| medication  | "No anti-epileptic drugs were prescribed to the patient" | "Anti-epileptic drugs were prescribed to the patient" |


*Table 1: Prompts used for zero-shot classification*
### Classification in a low-data regime
To further evaluate the generalization capability of the learned representations, we assess few-shot performance by training the classifier on a small subset, held out from the training of EEG-CLIP. The limited labeled data setting reflects realistic clinical scenarios where large labeled datasets are difficult to acquire. New clinical applications often only have access to small patient datasets. As such, assessing few-shot transfer is important for demonstrating clinical utility and feasibility.

## EEG data preprocessing
We preprocess the EEG data, taking inspiration from the preprocessing steps in [Schirrmeister et al., 2018](https://arxiv.org/abs/1703.05051) The following steps are applied to the EEG recordings in the TUAB dataset:

- Select a subset of 21 electrodes present in all recordings.
- Exclude the first minute of the recordings and only use the first 2 minutes after that.
- Clip the amplitude values to the range of ±800 \(\mu\)V to reduce the effects of strong artifacts.
- Resample the data to 100 Hz to further speed up the computation.
- Divide by 30 to get closer to unit variance.


## Architecture and training details

|![architecture](/assets/images/eegclip/eegclip_architecture.png)|
|:--:| 
| Architecture of EEG-CLIP |

For the EEG encoder $f_{\theta}$, we use a convolutional neural network (CNN) called Deep4 -([Schirrmeister et al., 2018](https://arxiv.org/abs/1703.05051)), whose architecture is optimized for the classification of EEG data. The Deep4 Network features four convolution-max-pooling blocks, using batch normalization and dropout, followed by a dense softmax classification layer. This enables the model to learn hierarchical spatial-temporal representations of the EEG signal. The output is flattened and passed to a fully-connected layer to derive a 128-dimensional embedding.

For the text encoder $g_{\phi}$, we leverage pretrained text encoders based on the BERT architecture ([Devin et al](https://arxiv.org/abs/1810.04805)). Such transformer-based models have shown state-of-the-art performance on a variety of natural language processing tasks. The advantage of these pretrained models is that they provide rich linguistic representations that can be effectively transferred to downstream tasks through fine-tuning.

The EEG and text embeddings are then fed into MLP projection heads, consisting of three fully-connected layers with ReLU activations. The final layer outputs a 64-dimensional projection of the embedding for contrastive learning. This architecture allows the model to learn alignments between EEG windows and corresponding medical report sentences in a shared embedding space. The contrastive loss enables the useful semantic features to be captured.

We train EEG-CLIP using the Adam optimizer with a learning rate of 5e-3 and weight decay of 5e-4. The model is trained for 20 epochs with a batch size of 64. We use the same training/testing split as in the TUAB dataset. Each recording is split into windows of length 1200, corresponding to a 12-second period, and with a stride of 519, which ensures all timesteps are predicted without any gap by our Deep4 model.

# Results and Discussion
### Classification

As shown in Table 2, on three of the four tasks, EEG-CLIP with a simple logistic regression classifier achieved strong performance, with balanced accuracy scores of 0.826 for pathological status, 0.687 for gender, and 0.713 for age. This indicates that the representations capture meaningful signal related to these key attributes.

| Task name   | EEG-CLIP + LogReg | EEG-CLIP + MLP | Task-specific model | Irrelevant task + MLP |
|-------------|-------------------|----------------|---------------------|----------------------|
| pathological| 0.826             | 0.847          | **0.851**           | 0.741 (age)          |
| gender      | 0.687             | 0.702          | **0.752**           | 0.667 (pathological) |
| age         | 0.713             | 0.747          | **0.786**           | 0.685 (pathological) |
| medication  | 0.633             | 0.615          | **0.685**           | 0.573 (pathological) |

*Table 2: Classification results (balanced accuracy on eval set)*

With a 3-layer MLP classifier head, performance improved further on all tasks, reaching 0.847, 0.702, and 0.747 for pathological status, gender, and age, respectively. The MLP can better exploit the relationships in the embedding space.

As expected, the task-specific models achieve the top scores, as they are optimized end-to-end directly on the evaluation data distribution and labels. However, EEG-CLIP comes reasonably close, especially with the MLP head, demonstrating the generalizability of the representations.

Compared to irrelevant task pretraining, EEG-CLIP substantially outperforms models pretrained on inconsistent targets like age or pathology. This confirms the importance of learning from the information contained in the medical reports.


### Zero-shot classification

| Task name   | Balanced accuracy on eval set |
|-------------|-------------------------------|
| pathological| 0.755                         |
| age         | 0.642                         |
| gender      | 0.567                         |
| medication  | 0.532                         |

*Table 3: Zero-shot classification results*

### Classification in a low-data regime

| Task name   | EEG-CLIP + MLP | Task-specific model | Irrelevant task + MLP |
|-------------|----------------|---------------------|----------------------|
| pathological| 0.710          | **0.781**           | 0.531 (age)          |
| gender      | 0.550          | **0.648**           | 0.512 (pathological) |
| age         | **0.712**      | 0.621               | 0.631 (pathological) |
| medication  | 0.551          | 0.575               | **0.598** (pathological) |

*Table 4: Classification results in low-data regime (balanced accuracy on eval set)*

On the pathological task, EEG-CLIP achieves 0.710 balanced accuracy on the held-out set. This approaches the 0.781 performance of a model trained from scratch with the same limited data. For age classification, EEG-CLIP even outperforms the specialized model. The medication task proves most challenging in the few-shot setting. However, all models struggle to exceed 0.6 accuracy, suggesting intrinsic difficulty of the binary prediction from small samples.

Critically, EEG-CLIP substantially outperforms models pretrained on irrelevant tasks across all but one experiment. This demonstrates the concrete value of pretraining on aligned data, even when fine-tuning data is scarce.


|![few_shot](/assets/images/eegclip/few_shot_results.png)|
|:--:| 
| * Few-shot accuracy for each task, for different training set sizes as fractions of the original training set size *|


Taken together, these quantitative results provide strong evidence for the quality and transferability of the multi-modal representations learned by EEG-CLIP. Performance across the range of evaluation paradigms demonstrates that it successfully encodes general semantic relationships between EEG and text. 

# Future Work and Conclusion
The EEG-CLIP framework presents a promising approach for learning versatile representations from EEG data by leveraging the information contained in medical reports. However, there are several potential avenues for future research to further improve and extend this framework.

One direction is to expand the training data by including more EEG recordings and their corresponding medical reports. As the EEG-CLIP framework relies on EEG-text pairs and does not require additional labeling, increasing the size and diversity of the training dataset could lead to more robust and generalizable representations. This is particularly advantageous since obtaining EEG-text pairs is relatively inexpensive compared to manually labeling EEG data for specific tasks.

Another exciting future direction is to explore the development of general-purpose pretrained EEG encoders using the EEG-CLIP framework. Similar to how pretrained language models like BERT have revolutionized natural language processing, pretrained EEG encoders could serve as a foundation for various EEG-related tasks. For example, pretrained encoders could be fine-tuned for specific applications such as motor task classification, emotion recognition, or sleep stage scoring, potentially reducing the need for large labeled datasets in each individual domain.

The significance of the EEG-CLIP framework lies in its ability to enable easier analyses of diverse decoding questions through zero-shot decoding or training task-specific models from fewer examples. By learning to align EEG signals with their textual descriptions, EEG-CLIP can perform classification tasks without explicit training, as demonstrated by the zero-shot classification results. Furthermore, the model's strong performance in low-data regimes highlights its potential for adapting to new tasks with limited labeled data, which is particularly valuable in medical research where obtaining large annotated datasets can be challenging.

In conclusion, the EEG-CLIP framework introduces a novel approach for learning versatile representations from EEG data by leveraging the information contained in medical reports through contrastive learning. The main contributions of this work include the development of a multimodal contrastive method that aligns EEG signals with their textual descriptions, the demonstration of its ability to outperform task-specific models in low-data regimes, and the showcase of zero-shot classification capabilities. The potential impact of EEG-CLIP on EEG analysis in medical research is significant, as it could facilitate the exploration of diverse decoding questions and enable more efficient development of EEG-based applications. With future extensions such as expanding the training data and building general-purpose pretrained EEG encoders, the EEG-CLIP framework has the potential to revolutionize the field of EEG analysis and contribute to advancements in neurological disorder diagnosis, brain-computer interfaces, and beyond