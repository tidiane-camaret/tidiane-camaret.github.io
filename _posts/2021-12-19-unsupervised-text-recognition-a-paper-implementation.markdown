---
layout: post
title:  "Unsupervised text recognition - a paper implementation"
date:   2021-12-20 11:20:25 +0100
categories: computer vision, unsupervised learning, text recognition
lang: en
ref: unsupervised learning
---

Text recognition is the task of **automatically retriving written text from images**. As humans, we are able to quickly identify written characters, but designing automatic methods achieving the same result is trickier : Same characters can be written with different fonts, sizes or colors. Some characters may also sometimes overlap each other, which makes individual character identification harder, and on top of that, images may be tainted with additional noise (paper texture, accidental ink stains ...) 

Most of the modern methods are flexible enough to tackle those challenges. However, they usually **require a large amount of labelled data** to be trained on, which means that in order to use them, we first need to gather thousands of text images of which we know the content beforehand. In some cases, such as low-ressource or extinct languages, this mandatory step is cumbersome.

We will here explore a method presented by **Gupta, Vedaldi, Zisserman** in their paper [Learning to read by spelling](https://arxiv.org/pdf/1809.08675.pdf), which achieves comparable results to modern methods, without relying on labeled data at all. 

We will first show the principles behind this method, and then try to reproduce those results on our own. [Our implementation can be found and tested here.](https://github.com/tidiane-camaret/read_by_spelling_impl)


# Text recogntion : The need of labelled data

We first can try to tackle the problem character by charater. This is actually of most of the earliest methods mostly tackled the problem, by trying to match each candidate with a given database of already-known characters. Obviously, this method is not very flexible, and we now use CNNS : they can accurately detect image features, and automatically pick which of those features are actually relevant for our problem. 

But to train our CNN, we need a dataset where each image is labeled.

It is actually often hard to treat this task at character level because characters are not easily separated. A more realistic dataset would be that each image is labeled as a whole, without indication of where the letters are : What we lose is letter alignment. 
We could train a CNN on each word/sentence of letters that we have a label for,but obviously we have less samples for each label, and we loose a lot of the CNN power.

There are some nice techniques to make the CNN work at characted level without letter alignment, such as CDC. Here we basically make probabilistic assumptions on where the separations are. We can train the CNN on the images zones that are the most susceptible to contain the letters, and the more we train, the better our assumptions are. This is a well-used method nowadays. But we still need a huge amount of labelled words/sentences.


The paper proposes an unsupervised approach, where only unlabeled data is be used: unlabeled images of text, and unrelated text data from the same language. 
This paper achieves ≈99% character accuracy and ≈95% word accuracy

# Unsupervised learning : overcoming the need of labeled data

In their paper, Gupta, Vedaldi and Zisserman propose an unsupervised approach, where only unlabeled data is used: unlabeled images of text, and unrelated text data from the same language.

One can ask : How can a model be trained on images without having access to their labels ? The answer is that the text examples already contain a fair amount of information on what a result text is supposed to look like, and this information can be exploited to separate and identify characters.

Let's imagine a text written in regular English, but where every letter of the alphabet has been replaced with an unknown symbol. 

At first glance, it would seem impossible to make sense of a text written with those symbols. However, we’re still able to extract some information from it : we can for example observe at which frequency each symbol/group of symbols appears, and at which positions.

It is then possible to make assumptions about which symbol corresponds to each character. The more text is available, the stronger assumptions we can make, and we can then associate each symbol to a known character of strongest probability. 

This task is also known as decipherment, and has taken a important part in the decoding of lost languages : https://pages.cs.wisc.edu/~bsnyder/papers/bsnyder_acl2010.pdf

The unsupervised text recognition problem can be separated into 3 sub-problems :
Character separation, where we separate each character in the image and associate them to an (unknown) label, 
Decipherment, where we link those labels to known characters. 


# character identification and cipher breaking : tackle two problems at the same time 
The problem is that those two problems cannot be tackled entirely separately.
We need to correctly identify characters to perform decipherment on them.

But in some cases, character identification may fail (two different letters can look similar, or the two same letters can be written slightly differently) 
In text recognition, those failures inevitably happen, and we need already decipherment assumptions to correct them.


So this is kind of a chicken and egg problem (decipherment needs recognition, and recognition requires decipherment), and the authors manage to tackle both of the problems at the same time by using adversarial training.


They design a model that inputs as text that is fed a text image and outputs a text, and another model that decides, given a text, whether the text makes sense or not. 

So at first, the first model, which is called a generator, outputs random letters, and it is easy for the second model (the discriminator), to tell apart real text from generated text.

The discriminator learns the statistical rules of what makes a real text. But the generator also learns from those rules and outputs more realistic text, and takes advantage of the input images to do so.





# tackle two problems at the same time ? - the use of adversarial training 

Principle :  The first part of the model is designed to propose a label for a given image. The second part is designed to decide whether a label belongs to real data or not.


# the proposed architecture 
