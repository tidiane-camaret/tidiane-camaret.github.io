---
layout: post
title:  "Unsupervised text recognition - a paper implementation"
date:   2021-12-20 11:20:25 +0100
categories: computer vision, unsupervised learning, text recognition
lang: en
ref: unsupervised learning
---

Text recognition is the task of **automatically retriving written text from images**. As humans, we are able to quickly identify written characters, but it can be tricky to design automatic methods achieving the same result : Indeed, the same characters can be written with different fonts, with a different size or color. On top of that, characters may sometimes overlap each other, or the images may be corrupted in some way. 

Most of the modern methods are flexible enough to tackle those challenges. However, they usually **require a large amount of labelled data** to be trained on, which means that in order to use them, we first need to gather thousands of text images of which we know the content beforehand. In some cases, such as low-ressource or extinct languages, this mandatory step is cumbersome.

We will here explore a method presented by **Gupta, Vedaldi, Zisserman** in their paper [Learning to read by spelling](https://arxiv.org/pdf/1809.08675.pdf), which achieves comparable results to modern methods, without relying on labeled data at all. 

We will first show the principles behind this method, and then try to reproduce those results on our own. [Our implementation can be found and tested here.](https://github.com/tidiane-camaret/read_by_spelling_impl)


# Text recogntion : The need of labelled data

The earliest methods mostly tackled the problem character-by-character, by matching a candidate with a given database of already-known characters.

The relatively recent democratization of CNNs has given a boost in text recognition : Not only are they able to accurately detect image features, but training them with data allows them to automatically pick which of those features are actually relevant for a given problem. 

However, this method also requires to gather a sufficient amount of data

# Unsupervised learning : How could a machine possibly read text when given text images only ? 



# How to identify each written character ?




# How to make sense of the identified characters ? - the use of adversarial training 



