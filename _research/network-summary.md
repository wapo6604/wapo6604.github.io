---
layout: page
title: Sampling Approaches to Graph Clustering and Evaluation on the StringDB PPI
permalink: /research/network-summ
header-includes:
   - \usepackage{amssymb, amsthm, amsmath, mathptmx, color, mathtools, bbm}
usemathjax: true
---

### Supervised by Prof Georg Gottwald ~ Dec 2024

This is a short summary and introduction to the work. For the full project, see [here]({{ site.baseurl }}/research/network-full).

### Table of Contents

1. [Background](#background)
    1. [Experimental Setup](#experimental-setup)
    2. [Hidden Markov Models](#hidden-markov-models)
2. [Methodology](#methodology)
3. [Our Findings](#our-findings)

### Background

As our best friends, we rely on dogs for some very important and strenuous tasks. For example, we have trained dogs to herd sheep and cattle, perform search and rescue operations, and even help in the military.


<figure>
  <img src="/_research/hmm-pics/slide1.png" alt="dogs in different circumstances" style="width:100%">
  <figcaption>Figure 1: Some examples of how dogs help us out.</figcaption>
</figure>

Especially in Australia however, where the climate is very hot, these tasks may put the dogs at a high risk of heat stress. This is a challenge for dog handlers, because working dogs, like Kelpies, are bred to be highly resilient and hardworking, so they do not exhibit warning signs until it becomes dangerous. Moreover, different dogs, even of the same breed, and under the same environmental conditions, may have very different responses. Of course, dogs may be working out of the sight of their handlers as well. 

These factors make it very hard for handlers to judge a dog's wellbeing by qualitative means. For this reason, it is useful to be able to detect the signs of overtemperature stress using a quantitative approach.


#### Experimental Setup

In order to test any machine learning algorithm however, we need some data. To collect it, we fitted six kelpies with a harness recording their ECG patterns, respiratory excursions, acceleration and temperature at each dog's back and belly. Each dog also ingested a pill that measured their internal body temperature.

<figure>
  <img src="/_research/hmm-pics/slide3.png" alt="pills and harness" style="width:100%">
  <figcaption>Figure 2: Left: kelpies wearing sensor harness, right: temperature sensing pill.</figcaption>
</figure>

The dogs where then given a set of activities to perform throughout the day. The graphs below show the data collected from the harness of the dog Bobby,

<figure>
  <img src="/_research/hmm-pics/datamv2.png" alt="harness data" style="width:100%">
  <figcaption>Figure 3: Harness data from Bobby.</figcaption>
</figure>

and also the data collected from Bobby's temperature sensing pill. 

<figure>
  <img src="/_research/hmm-pics/datauv.png" alt="pill data" style="width:100%">
  <figcaption>Figure 4: Pill data from Bobby.</figcaption>
</figure>

From this data then, we want to be able to infer the behavioural stress of the dog, and in particular, whether the dog is in a high activity state and hence at a heightened risk of heat stress.

#### Hidden Markov Models

First, we introduce some terminology. Represent the current behavioural state of the dog at time $t$ out of $N$ possible states by the random variable $S_t\in \{1,\dotsb, N\}$.
        
Then we say that the state sequence $S_{1:T} = (S_1,\dotsb,S_T)$ satisfies the *Markov Property* if

$$ P(S_t\mid S_{1:(t-1)}) = P(S_t \mid S_{t-1})\,. $$

We will associate with the state sequence $S_{1:T}$ with an *initial probabilites vector*

$$\boldsymbol{\theta}_{\text{init}} = \big [ P(S_1 = 1), P(S_1=2), \dots, P(S_1=N)\big ] \,.$$

which encodes for the initial behavioural state of the dog, and a *transition probabilities matrix*: a matrix $\boldsymbol{\theta}\_{\text{trans}_{N\times N}}$ such that 

$$\boldsymbol{\theta}_{\text{trans}_{i,j}} = P(S_t = i \mid S_{t-1}= j)$$\,.

The transition probabilities matrix encodes for how the dog moves between different behavioural states.

Now, represent the observation at time $t$ by the random variable $X_t$, and associate with $X_{1:T}=(X_1,\dots,X_T)$ some distribution parameters $\boldsymbol{\theta}\_{\text{obs}}$. For example, we might say $X_1\sim\mathrm{Normal}(0,1)$ and $X_2\sim \mathrm{Normal}(2,3)$, so these pairs of mean and variances would be stored in $\boldsymbol{\theta}\_{\text{obs}}$.

Note that $X_t$ could be a scalar or a random vector as well. 

