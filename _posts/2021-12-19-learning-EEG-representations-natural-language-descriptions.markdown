---
layout: post
title:  "EEG-Clip : Learning EEG representations from natural language descriptions"
date:   2024-03-28 11:20:25 +0100
categories: computer vision, constrastive learning, text processing, EEG
lang: en
ref: unsupervised learning
---

Recent advances in machine learning have led to deep neural networks being commonly applied to electroencephalogram (EEG) data for a variety of decoding tasks. EEG is a non-invasive method that records the electrical activity of the brain using electrodes placed on the scalp. While deep learning models can achieve state-of-the-art performance on specialized EEG tasks, most EEG analyses focus on training task-specific models for one type of classification or regression problem [Heilmeyer et al., 2018].

However, medical EEG recordings are often annotated with additional unstructured data in the form of free text reports written by neurologists and medical experts that can be exploited as a source of supervision. In the computer vision domain, Contrastive Language–Image Pre-training [Radford et al., 2021] (CLIP) leverages this text-image pairing to learn visual representations that effectively transfer across tasks.

Inspired by CLIP, we propose EEG-Clip - a contrastive learning approach to align EEG time series data with corresponding clinical text descriptions in a shared embedding space. This work explores two central questions: (i) how textual reports can be incorporated into an EEG training pipeline, and (ii) to what extent this multimodal approach contributes to more general EEG representation learning.

We demonstrate EEG-CLIP's potential for versatile few-shot and zero-shot EEG decoding across multiple tasks and datasets. EEG-CLIP achieves nontrivial zero-shot classification results. Our few-shot results show gains over previous transfer learning techniques and task-specific models in low-data regimes. This presents a promising approach to enable easier analyses of diverse decoding questions through zero-shot decoding or training task-specific models from fewer training examples, potentially facilitating EEG analysis in medical research.

# Methodology

Contrastive self-supervised learning has recently emerged as a powerful approach for learning general visual representations. Models like CLIP [Radford et al., 2021] are trained to align image $x_i$ and corresponding text $y_i$ embeddings by minimizing a contrastive loss $\mathcal{L}$:

$$\mathcal{L}=\sum_{i=1}^{N}-\log \frac{\exp(\text{sim}(x_i,y_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(x_i,y_j)/\tau)}$$

where sim(.) is a measure of similarity, such as cosine similarity, and $\tau$ is a temperature parameter that controls the softness of the distribution. This objective brings matching image-text pairs closer and separates mismatched pairs in the learned embedding space.

CLIP is trained on a vast dataset of 400 million image-text pairs from diverse Internet sources with unstructured annotations. Through this natural language supervision, CLIP develops versatile image representations that achieve strong zero-shot inference on downstream tasks by querying the aligned embedding space.

Inspired by the success of CLIP, we propose EEG-CLIP, a contrastive learning framework for aligning EEG time series data with corresponding clinical text descriptions. EEG-CLIP consists of two encoder networks: an EEG encoder $f_{\theta}$ and a text encoder $g_{\phi}$. The EEG encoder $f_{\theta}$ maps the input EEG time series $x_i$ to a fixed-dimensional embedding vector $f_{\theta}(x_i)$, while the text encoder $g_{\phi}$ maps the corresponding clinical text description $y_i$ to an embedding vector $g_{\phi}(y_i)$.

The encoders are trained to minimize the contrastive loss $\mathcal{L}$, which encourages the embeddings of matching EEG-text pairs to be similar while pushing apart the embeddings of mismatched pairs. The similarity measure sim(.) used in EEG-CLIP is the cosine similarity between the normalized embeddings:

$$\text{sim}(x_i, y_j) = \frac{f_{\theta}(x_i)^\top g_{\phi}(y_j)}{|f_{\theta}(x_i)| |g_{\phi}(y_j)|}$$

During training, EEG-CLIP learns to align the EEG and text embeddings in a shared space. This alignment enables versatile EEG decoding tasks, such as zero-shot classification, where the model can predict the class of an unseen EEG sample by comparing its embedding with the embeddings of textual class descriptions.

# Experimental Setup
   - Describe the dataset used for training and evaluation, including the number of EEG recordings and their corresponding medical reports.
   - Explain the few-shot and zero-shot settings used to assess the performance of EEG-CLIP.
   - Mention any data preprocessing steps or specific evaluation metrics used.

# Results and Discussion
   - Present the main findings from your experiments, highlighting the performance of EEG-CLIP in various few-shot and zero-shot settings.
   - Discuss the implications of your results, emphasizing the potential of EEG-CLIP for versatile EEG decoding.
   - Compare your approach with existing methods and discuss any advantages or limitations.

# Future Work and Conclusion
   - Discuss potential future directions for improving and extending the EEG-CLIP framework.
   - Highlight the significance of your work in enabling easier analyses of diverse decoding questions through zero-shot decoding or training task-specific models from fewer examples.
   - Conclude by summarizing the main contributions of your work and its potential impact on EEG analysis in medical research.


![img1](/assets/images/unsupervised_text_recongition/im1.png)

As humans, we are able to quickly identify written characters, but designing automatic methods achieving the same result is trickier : Same characters can be written with different fonts, sizes or colors. Some characters may also sometimes overlap each other, which makes individual character identification harder, and on top of that, images may be tainted with additional noise (paper texture, accidental ink stains ...) 

![img2](/assets/images/unsupervised_text_recognition/im2.png)

Most of the modern methods are flexible enough to tackle those challenges. However, they usually **require a large amount of labelled data** to be trained on, which means that in order to use them, we first need to gather thousands of text images of which we know the content beforehand. In some cases, such as low-ressource or extinct languages, this mandatory step is cumbersome.

We will here explore a method presented by **Gupta, Vedaldi, Zisserman** in their paper [Learning to read by spelling](https://arxiv.org/pdf/1809.08675.pdf), which achieves comparable results to modern methods, without relying on labeled data at all. 

We will first show the principles behind this method, and then try to reproduce those results on our own. [Our implementation can be found and tested here.](https://github.com/tidiane-camaret/read_by_spelling_impl)

