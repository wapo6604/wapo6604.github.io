---
layout: page
title: Consistency Results on Estimating the Number of Clusters
permalink: /research/pitman-full
header-includes:
   - \usepackage{amssymb, amsthm, amsmath, mathptmx, color, mathtools, bbm}
usemathjax: true
---

### Supervised by A/Prof Clara Grazian ~ Dec 2024

For a summary of the project, see [here]({{ site.baseurl }}/research/pitman-summ).

$$\newcommand{\dir}[0]{\mathrm{Dirichlet}}$$
$$\newcommand{\be}[0]{\mathrm{Beta}}$$
$$\newcommand{\mult}[0]{\mathrm{Multinomial}}$$
$$\newcommand{\pr}[0]{\mathrm{pr}}$$
$$\newcommand{\iid}[0]{\overset{\mathrm{iid}}{\sim}}$$
$$\newcommand{\ind}[0]{\overset{\mathrm{ind}}{\sim}}$$



### Abstract
We review the literature on consistency results for estimating the number of components in a mixture model with Dirichlet and Pitman-Yor process models. Then we show that Pitman-Yor process models with a uniform prior for the concentration parameter can be a consistent estimator for the number of components in a mixture model under some conditions. 

### Table of Contents

1. [Introduction](#introduction)
2. [Aims of the Paper](#aims)
3. [Methods in the literature](#methods)  
    1. [Inconsistency result for fixed parameters](#inconsistency-result-for-fixed-parameters)
    2. [Consistency Result with a Prior](#consistency-result-with-a-prior)
    3. [Required Steps for our Extension](#required-steps-for-our-extension)
4. [Results](#results)    
    1. [New Bound on $R(n,t,s)$](#new-bound-on-rnts)  
    2. [New Bound on $C(n,t,s)$](#new-bound-on-cnts)
5. [Discussion](#discussion)
6. [Insights](#insights)
7. [Future Work](#future-work)
8. [Conclusions](#conclusion)
9. [References](#references)

### Introduction

Suppose we collect some data from a population in which we believe there exist distinct subpopulations that respond differently to the study. A sensible way to model such data might be to assume that the observations sampled from each subpopulation have their own distribution. 

This structure is naturally encoded by a mixture model. Suppose there are $t$ subpopulations, where the data sampled from subpopulation $j$ is $R_j$ distributed for each $j=1,\dots, t$. If we further suppose that the probability of sampling data from subpopulation $j$ is $p_j$, so that $\sum_{j=1}^t p_j= 1$, we can define a data generating mechanism where each observation $X_n$ is distributed according to 
\begin{equation}
    X_n \sim P = \sum_{j=1}^t p_j R_j \,.
\end{equation}

Call the contribution of each subpopulation a *mixture component*. 
If we specify the number of mixture components $t$ and the component distributions $R_j$, then there are many ways of fitting the required parameters to the observed data, both Bayesian and frequentist. 
Moreover, we can use the fitted model to cluster the data by allocating each observation to the mixture component which maximises the likelihood. However, a challenge with this analysis is choosing the number of mixture components to use in the mixture model or, alternatively, the number of clusters to fit.  

One way of avoiding this choice is to use a Dirichlet process to estimate the number of components (Pella and Masuda, 2006). 
Following Ferguson (1973), let $Q_0$ be a non-null finite measure on $(X,\mathcal A)$ and $\alpha>0$. Call $\mathrm{DP}(\alpha, Q_0)$ a Dirichlet process on $(X,\mathcal A)$ if for every $k=1,2, \dots$ and a measurable partition $(B_1,\dots, B_k)$ of $X$, the distribution of $(Q_0(B_1),\dots, Q_0(B_k))$ is $\dir(\alpha Q_0(B_1) , \dots, \alpha Q_0(B_k))$. Then, we can define a Dirichlet process mixture model, essentially an infinite mixture model, by drawing our data according to

\begin{equation}\label{eq:dpm}
     X_i \mid \theta_i \ind k(\cdot \mid \theta_i),  \quad\quad
    \theta_i \mid \widetilde{P} \iid \widetilde{P}, \quad\quad 
    \widetilde{P} \sim \mathrm{DP}(\alpha, Q_0) \,,   
\end{equation}

where each mixture component has density $k(\cdot | \theta_i)$ for some component parameters $\theta_i$.
In essence, we draw a discrete probability distribution $\widetilde{P}$ from the Dirichlet process for our component parameters, then draw our component parameters from $\widetilde{P}$. Finally, we sample our data points $X_i$ from an infinite mixture of densities $k(\cdot \mid \theta_i)$. In the posterior, only finitely many of these components will have mass, hence choosing the number of components in the mixture model implicitly. 

A useful property of the Dirichlet process is that clusters tend to be allocated according to a rich-gets-richer scheme, a result of the Bayes property of the Dirichlet distribution (Ferguson, 1973). In particular, for a sample $X_1,\dots, X_n$ generated from a Dirichlet process $\mathrm{DP}(\alpha,Q_0)$, if $c_k$ represents the component membership of $X_k$, and $\mathbf{c}_{-k}$ is the component memberships of all other observations, then 

$$ \mathbb{P}(c_k = i \mid \mathbf{c}_{-k} ,\alpha) = \frac{n_i}{n-1+\alpha} \,, $$

where $n_i$ is the number of members in the $i$-th cluster, excluding possibly the $k$-th observation. Moreover, if there are $t$ components with data points, then the probability of an observation being sampled from a new component would be 

$$ 1- \sum_{i=1}^t\mathbb{P}(c_k = i \mid \mathbf{c}_{-k} ,\alpha) = \frac{\alpha}{n-1+\alpha} \,. $$

The Pitman-Yor process extends the Dirichlet process and provides more control over the cluster sizes (Pitman and Yor, 1997). It has three parameters: a base distribution $Q_0$, a discount parameter $0\leq d<1$ and a concentration parameter $\alpha > -d$. In a Pitman-Yor $\mathrm{PY}(\alpha, d, Q_0)$ process, the component membership probabilities instead satisfy

$$ \mathbb{P}(c_k = i \mid \mathbf{c}_{-k} ,\alpha, d)  = \frac{n_i- d}{n+\alpha} \quad \text{and} \quad  1- \sum_{i=1}^t\mathbb{P}(c_k = i \mid \mathbf{c}_{-k} ,\alpha) = \frac{td + \alpha}{n+\alpha} \,, $$

noting that when $d=0$, the Pitman-Yor process collapses to a Dirichlet process (Aldous, 1985). In expectation, the cluster sizes of the Pitman-Yor process tend to follow a power-law, while the Dirichlet process follows an exponential distribution (Pitman, 2002). 

Both Dirichlet and Pitman-Yor process models are commonly used in the literature to estimate the number of mixture components (Ishwaran and James, 2001). However, it was shown by Miller and Harrison (2014) that estimating the number of components using Pitman-Yor $\mathrm{PY}(\alpha, d,Q_0)$ models (and hence also Dirichlet process models) with a fixed concentration $\alpha$ and discount $d$, is in fact inconsistent. The heuristic idea is that as the number of data points increases, the penalty for creating new clusters is not great enough, so very small transient clusters are formed to accommodate outlier points. Hence, the posterior will favour a greater number of components than the true value (Onogi et al., 2011).

Since the probability of cluster formation depends only on $\alpha$, many practical implementations of Dirichlet process models also specify a prior $\pi$ for $\alpha$ to get better flexibility. Contrary to the results of Miller and Harrison, under some regularity conditions on the prior $\pi$, Ascolani et al. (2023) were able to show that estimating the number of clusters with a Dirichlet process mixture model of the form

\begin{equation}\label{eq:pym0}
     X_i \mid \theta_i \ind k(\cdot \mid \theta_i),  \quad\quad
    \theta_i \mid \widetilde{P} \iid \widetilde{P}, \quad\quad 
    \widetilde{P} \sim \mathrm{DP}(\alpha, Q_0) \,, \quad \quad \alpha\mid \pi\,.
\end{equation}
attains consistency.

### Aims

Naturally, we ask whether it is possible to extend the consistency results of Ascolani et al. to Pitman-Yor process models of the form

\begin{equation}\label{eq:pym}
     X_i \mid \theta_i \ind k(\cdot \mid \theta_i),  \quad\quad
    \theta_i \mid \widetilde{P} \iid \widetilde{P}, \quad\quad 
    \widetilde{P} \sim \mathrm{PY}(\alpha, d, Q_0) \,, \quad \quad \alpha\mid \pi\,.
\end{equation}

where the discount $d$ is held constant. To do this, we first aim to explore the inconsistency result of Miller and Harrison and understand why the likelihood is not significantly decreased by the formation of small clusters. 

We then explore the positive consistency result of Ascolani et al. with the aim of understanding how the regularity conditions imposed on the prior $\pi$ change the clustering behaviour.

To extend the results of Ascolani et al., we note that the Dirichlet process model favours a smaller number of clusters than the Pitman-Yor model. Therefore, we will need to explore stronger regularity conditions on the prior $\pi$ for the concentration $\alpha$.

### Methods

Since we are interested in the distribution of the number of mixture components, it is convenient to rewrite the models in  (\ref{eq:dpm}), (\ref{eq:pym0}) and (\ref{eq:pym}) over a distribution of partitions to represent the clusters generated by the model. 

Fortunately, Pitman (1994) provides such a representation. Denote the set of unordered partitions of $\{ 1,\dots,n\}$ into $s$ components (where $s \leq n$) by $\tau_s(n)$. Then for a random partition $\mathbf A := (A_1,\dots, A_s) \in \tau_s(n)$, the sequence of component parameters $\theta_i$ induces the prior

\begin{equation}\label{eq:pympd}
\mathbb{P}(\mathbf A \mid \alpha, d) :=  \mathbb{P}(\|A_1\|,\dots,\|A_s\| \mid \alpha, d) = \dfrac{(\alpha + d)\_{s-1 \uparrow d}}{(\alpha+1)\_{n-1\uparrow 1}} \prod\_{i=1}^s(1-d)_{\|A_i\|-1 \uparrow 1} 
\end{equation}

for a Pitman-Yor $\mathrm{PY}(\alpha, d, Q_0)$ process, where $x_{n\uparrow \delta} := x(x+\delta)\cdots (x+(n-1)\delta)$ is the rising factorial. We call this representation the *partition distribution* of the Pitman-Yor process. In particular, the Dirichlet process $\mathrm{DP}(\alpha, Q_0)$ corresponds to the Pitman-Yor process $\mathrm{PY}(\alpha, 0, Q_0)$, so the partition distribution of the Dirichlet process is

\begin{equation}\label{eq:dppd}
\mathbb P(\mathbf A \mid \alpha) := \mathbb P(\|A_1\|,\dots,\|A_s\|\mid \alpha) = \frac{\alpha^s}{(\alpha+1)\_{n-1\uparrow 1}} \prod_{i=1}^s (\|A_i\|-1)! \,.
\end{equation}

Conditional on a clustering $\mathbf A \in \tau_s(n)$, we can specify a probability distribution on our cluster parameters $\hat{\theta}_{1:s} := (\hat{\theta}_1,\dots, \hat{\theta}_s)$ as

\begin{equation}
    \mathbb P(\hat{\theta}\_{1:s} \mid \mathbf A , \alpha) = \prod\_{i=1}^s q\_0(\hat{\theta}\_i)\,,
\end{equation}

where $q_0$ is the density of $Q_0$. We can also specify a probability distribution over the data $X_{1:n}:= (X_1,\dots,X_n)$ by

\begin{equation}
    \mathbb P(X\_{1:n} \mid \hat{\theta}\_{1:s}, \mathbf A) = \prod_{j=1}^s \prod\_{i\in A\_j} k(X\_i\mid \hat{\theta}\_j)\,.
\end{equation}

As introduced in Miller and Harrison, for observations $X_{1:n}$, and a subset of $A_j \subseteq \{1,\dots, n\}$ with a corresponding cluster parameter $\theta$ in parameter space $\Theta$, we define the *single-cluster marginal*

\begin{equation}
    m(X_{A_j}) = \int_{\Theta} \prod_{j\in A_j} k(x_j \mid \theta) q_0(\theta) d\theta
\end{equation}

which represents the likelihood of the observations in $A_j$, given that they are in the same cluster.

#### Inconsistency Result For Fixed Parameters
Miller and Harrison were able to prove that, for fixed parameters $0\leq d < 1$ and $\alpha>-d$, the Pitman-Yor process model is an inconsistent estimator of the number of mixture components. To do this, they needed two main conditions.

The first considers the partition distribution of the clustering process. For a partition 
$\mathbf A\in \mathcal \tau_t(n)$, let $R_{\mathbf A} := \bigcup_{\|A_i\|\geq 2} A_i$ and for $j\in R_{\mathbf A}$, let $\mathbf B(\mathbf A,j)$ be the partition $\mathbf B\in \mathcal \tau_{t+1}(n)$ such that

$$B_i = A_i\setminus \{j\} \quad \text{ and } \quad B_{t+1} = \{j\} \,,$$

that is, remove $j$ from the cluster it belongs to in $\mathbf A$, and add it to its own cluster. 
Then, let $\mathcal Z_{\mathbf A}$ be the collection $\{\mathbf B(\mathbf A,j) : j\in R_{\mathbf A} \}$. For $1\leq t < n$, let 

\begin{equation}\label{eq:cnt}
    c_n(t) = \frac1n \max_{\mathbf A\in \tau_t(n)} \max_{\mathbf B\in \mathcal Z_{ \mathbf A}} \frac{ \mathbb P (\mathbf A)}{ \mathbb P (\mathbf B)} \,.
\end{equation}

This quantity is large if there is a partition $\mathbf A\in \mathcal \tau_t(n)$ such that taking out some element $j$ and moving it to its own cluster creates a new partition $\mathbf B \in \tau_{t+1}(n)$ with $\mathbb P(\mathbf B)$ much smaller than $\mathbb P(\mathbf A)$. This behaviour is good because it shows the model can settle on some number of clusters. If, instead, we have the condition that

\begin{equation}\label{eq:cond3}
    \limsup_{n\to \infty} c_n(t) <\infty
\end{equation} 

for some $t$, then $\mathbb P(\mathbf B)$ never disappears compared to $\mathbb P(\mathbf A)$, and spurious clusters can always be formed. In particular, Miller and Harrison proved that (\ref{eq:cond3}) is true for every $t$ for the Pitman-Yor partition distribution.

The second condition characterises the behaviour of the single-cluster marginals. For $1\leq t\leq n$, observations $X_{1:n}$, and a constant $C\geq 0$, let

\begin{equation}\label{eq:vt}
    \varphi_t(X_{1:n},C) = \min_{\mathbf A\in \tau_t(n)}\frac1n| S_{\mathbf A}(X_{1:n},C)| 
\end{equation}

where $S_{\mathbf A}(X_{1:n},C)$ is the set of indices $j\in \{1,\dots, n\}$ such that the part $A_l \in \mathbf A$ containing index $j$ satisfies 

\begin{equation}
    m(x_{A_l})\leq Cm(X_{A_{l}\setminus \{j\}})m(X_{\{j\}}) \,.
\end{equation}

For some number of components $t$, we need to show that the likelihood of the refinement $\mathbb P(X_{1:n}\mid \mathbf B)$ does not disappear compared to $\mathbb P(X_{1:n}\mid \mathbf A)$. This is encoded by the second condition of Miller and Harrison, which is given by

\begin{equation}\label{eq:cond4}
    \sup_{C\geq 0} \liminf_{n\to\infty} \varphi_t(X_{1:n},C)>0 \,.
\end{equation}

This condition is also satisfied by the single-cluster marginals of the Pitman-Yor process.


If a clustering process satisfies both conditions, i.e. (\ref{eq:cond3}) and (\ref{eq:cond4}), then Miller and Harrison were able to show that the posterior on the number of clusters, $K_n$,  satisfies

\begin{equation}
    \limsup_{n\to \infty} P(K_n =t\mid X_{1:n})<1 \quad \text{ almost surely}\,,
\end{equation}

for any number of clusters $t$. In particular, the Pitman-Yor process with fixed parameters satisfies both conditions, so it cannot be a consistent estimator for the number of clusters in a mixture model. 


#### Consistency Result with a Prior

However, Ascolani et al. were able to show that the Dirichlet process model given in (\ref{eq:pym0}) is consistent when a prior $\pi$ is introduced for $\alpha$. To represent this, let $X_{1:n}$ be random variables representing our observations and $K_n$ the estimator for the number of clusters. Then we have

\begin{equation}
    \mathbb P(X_{1:n} = x_{1:n}, K_n = s) = \sum_{\mathbf A\in \tau_s(n)} \mathbb P(\mathbf A) \prod_{j=1}^s m(X_{A_j}) 
\end{equation}

where $\mathbb  P(\mathbf A) = \int_{\mathbb R} \mathbb P(\mathbf A\mid \alpha)\pi(\alpha)d\alpha$ and $\mathbb P(\mathbf A\mid \alpha)$ is the partition distribution for the Dirichlet process in (\ref{eq:dppd}). Hence, if $t$ is the true number of mixture components, we want to show the  posterior

<div class="scrollable-equation">
\begin{equation}\label{eq:post}
    \mathbb P(K_n = t\mid X_{1:n}) = \frac{1}{\mathbb P(X_{1:n})} \int_{\mathbb R} \frac{\alpha^t}{(\alpha+1)\_{n-1 \uparrow 1}} \pi(\alpha) d\alpha \cdot \sum_{\mathbf A\in \tau_s(n)} \prod_{j=1}^s (|A_j|-1)!\prod_{j=1}^s m(X_{A_j})  \to 1
\end{equation}
</div>

as $n\to \infty$ almost surely. To do this, Ascolani et al. needed to place some assumptions on the prior:

- **A1.**  Prior $\pi$ is absolutely continuous with respect to the Lebesgue measure.
- **A2.**  (Polynomial behaviour). There exists $\varepsilon,\delta, \beta$ such that for all $\alpha\in(0,\varepsilon)$ we have
    
\begin{equation} 
\frac{\alpha^{\beta}}{\delta} \leq \pi(\alpha) \leq \delta \alpha^{\beta} \,.
\end{equation}

- **A3.** (Subfactorial Moments). There is $D,\nu,\rho >0 $ such that, for all $s\geq 1$, 

\begin{equation}\label{eq:subfact}
    \int_{\mathbb R} \alpha^s \pi(\alpha) d\alpha < D\rho^{-s} \Gamma(\nu + s + 1) \,.
\end{equation}

Note that if these constants exist, then by changing them slightly, we may satisfy (\ref{eq:subfact}) with an arbitrarily large $\rho$ since factorial growth is stronger than exponential decay.


We also impose some assumptions on the data-generating densities $k(\cdot \mid \theta)$. In particular, Ascolani et al. assumes that each $\theta\in\mathbb R$ acts like a location parameter, and so the densities can be rewritten as

\begin{equation}
    k(x\mid \theta)  = g( x - \theta)
\end{equation} 

where $g$ is a density in $\mathbb R$ such that there is some interval $[a,b]$ on which $g>0$ and $g=0$ otherwise, and $g$ is differentiable and has a bounded derivative in $(a,b)$. They also require that the base measure $Q_0$ for the Dirichlet process is absolutely continuous with respect to the Lebesgue measure and $q_0$ is bounded. Finally, Ascolani et al. also assume that the true cluster parameters $\theta^\*$ are *completely separated*, that is, 

\begin{equation}
  |\theta_j^\*-\theta_k^\*| > b-a \quad \text{for all} \quad j\neq k   \,.
\end{equation}

Under these assumptions, if $t$ is the true number of clusters, and $X_{1:n}$ are our data points, then the main idea of Ascolani et al. is to study whether we have

\begin{equation}\label{eq:tozero}
    \sum_{s\neq t} \frac{\mathbb P(K_n = s \mid X_{1:n})}{\mathbb P (K_n = t\mid X_{1:n})} \to 0 \quad \text{ as } \quad n\to\infty  \quad \text{ almost surely }
\end{equation}

because  

\begin{equation}
    \mathbb P(K_n = t \mid X_{1:n}) = \left[ 1 + \sum_{s\neq t} \frac{\mathbb P(K_n = s \mid X_{1:n})}{\mathbb P (K_n = t\mid X_{1:n})}  \right]^{-1} \,,
\end{equation}

so if (\ref{eq:tozero}) is true then the posterior $\mathbb P(K_n = t \mid X_{1:n}) \to 1$ as $n\to\infty$ almost surely as well. To study the convergence of this series, Ascolani et al. writes each ratio of probabilities as the product

<div class="scrollable-equation">
\begin{equation}
    \frac{\mathbb P(K_n = s \mid X_{1:n})}{\mathbb P (K_n = t\mid X_{1:n})} =  \underbrace{\frac{\int_{\mathbb R} \frac{\alpha^s}{(\alpha+1)\_{n-1 \uparrow 1}} \pi(\alpha) d\alpha }{   \int_{\mathbb R} \frac{\alpha^t}{(\alpha + 1)\_{n -1 \uparrow 1}}  \pi(\alpha) d\alpha }}\_{C(n,t,s)} \cdot \underbrace{\frac{\sum_{\mathbf A\in\tau_{s}(n) }\left[ \prod_{j=1}^s (|A_j|-1)! \prod_{j=1}^s m(X_{A_j})\right] }{  \sum_{\mathbf B\in\tau_{t}(n) }\left[ \prod_{j=1}^t (|B_j|-1)! \prod_{j=1}^t m(X_{B_j})\right]}}_{R(n,t,s)}
\end{equation}
</div>

and bounds each part appropriately. This generalises Miller and Harrison's work -- where the $c_n(t)$ term in (\ref{eq:cnt}) encodes the ratio of the probabilities for partition $\mathbf A\in\tau_t(n)$ and its refinement $\mathbf B$ into $t+1$ parts, the $C(n,t,s)$ the term represents the partition probability ratio for any further refinement into $s$ parts. Moreover, the $R(n,t,s)$ term generalises the $\varphi_t$ term in (\ref{eq:vt}) for the same purpose. 

The main bound for $C(n,t,s)$ is given by Corollary 1 of Ascolani et al. and states that if $\pi$ satisfies A1 and A2, then for a constant $G$ 
depending on $\delta$ and $\varepsilon$ from A2, we have for every $0<s<n$ and $n\geq 4$ 

\begin{equation}\label{eq:c1}
    C(n,t,t+s) \leq \frac{G\Gamma(t  +\beta + 1)2^s s }{\varepsilon\log\left(\frac{n}{1+\varepsilon}\right)}\mathbb E(\alpha^{t+s-1}) \,.
\end{equation}

The bound for $R(n,t,s)$ is trickier.
Since the true cluster parameters $\mathbf \theta_{1:t}^\* := (\theta_1^\*,\dots, \theta_t^\*)$ are completely separated, each $x$ can have a non-zero density for at most one component of the mixture.  With this in mind, define sets of the form 

\begin{equation}
    C\_j = \{ i\in \{1,\dots, n\} : x\_i \in [\theta_j^\* + a, \theta_j^\* +b]\}
\end{equation}

and call $n\_j := \|C\_j\|$. These are the data points with positive density under the $\theta\_j^\*$ component density, and these sets are disjoint by the completely separated property. In particular, $\bigcup\_{j=1}^t C\_j = \{ 1,2\dots,n\}$ so  $\sum\_{j=1}^t n\_j = n$.

With this notation, we can rewrite $R(n,t,s)$ in the form

<div class="scrollable-equation">
$$
\begin{equation}\label{eq:rnts}
     R(n,t,s) = \sum_{\mathbf s \in \mathbf S} \prod_{j=1}^t \sum_{\mathbf A_j \in \tau_{s_j}(n_j)} \frac{\prod_{k=1}^{s_j} (|A_k^j| -1)!}{(n_j -1)!} \frac{\prod_{k=1}^{s_j} \int_{\mathbb R} \prod_{i\in A_{k}^j} \frac{g(X_i-\theta_k)}{p_j g(X_i - \theta_j^*)} q_0(\theta_j) d\theta_j } {  \int_{\mathbb R} \prod_{i\in C_j} \frac{g(X_i-\theta_k)}{p_j g(X_i - \theta_j^*)} q_0(\theta_j) d\theta_j} \,,
\end{equation}
$$
</div>

where $\mathbf S$ is the set of simplices 

$$ \mathbf S =\left\{(s_1,\dots,s_t) \text{ where } s_j \leq n_j \text{ and } \sum_{j=1}^t s_j = s\right\} \,.$$

The key bounds on $R(n,t,s)$ are established in Ascolani et al. by Lemmas 11 to 14. In Lemma 11, we have that for each $j=1,\dots, t$, there is $K>0$ and $N_j\in \mathbb N$ such that for all $n_j \geq N_j$, 

\begin{equation}
\int_{\mathbb R} \prod_{i\in C_j} \frac{g(X_i -\theta_j)}{g(X_i-\theta_j^*)} q_0(\theta_j) d\theta_j \geq \frac{K^{\frac1t} Y_{n_j}^j }{ n_j (\log n )^{\frac{1}{2t}}} \,.
\end{equation}

Under Lemma 12, we have that for each $j=1,\dots, t$ and $s_j\geq 1$ and $(\theta_1,\dots, \theta\_{s_j})\in \mathbb R^{s_j}$, 

\begin{equation}\label{eq:lemma12}
\mathbb E\left[ \prod_{k=1}^{s_j} \int_{\mathbb R^{s_j}}\prod_{i\in A_k^j} \frac{g(X_i-\theta_h)}{g(X_i -\theta_j^*)} q_0(\theta_h) d\theta_h\right] \leq \left(\frac{U}{m}\right)^{s_j} \prod_{k=1}^{s_j}\frac{1}{|A_k^j|+ 1}
\end{equation}

where $U,m>0$ are constants that depend on the base distribution $Q\_0$ and density $g$ respectively. 
If we apply these to (\ref{eq:rnts}) and factor out the common terms, we obtain 

<div class="scrollable-equation">
\begin{equation}
      \mathbb E [R(n,t,s)] \leq \frac{2^t \sqrt{\log n}}{K} \left[\frac Um\right]^s \sum_{\mathbf s\in \mathbf S} \prod_{j=1}^t \sum_{\mathbf A_j \in \tau_{s_j} (n_j)} \frac{n_j\prod_{k=1}^{s_j} (|A_k^j| -1)!}{(n_j-1)!} \cdot \frac{1}{\prod_{k=1}^{s_j} (|A_k^j|+1)}\,.
\end{equation}
</div>

Also, Lemma 3 of Ascolani et al. states that we have

\begin{equation}
    \sum_{\mathbf A_j\in \tau_{s_j}(n_j)} \frac{\prod_{k=1}^{s_j} (|A_k^j|-1)!}{(n_j-1)!} = \sum_{\mathbf a_j\in \mathcal F_{s_j}(n_j)} \frac{n_j}{s_j!\prod_{k=1}^s a_k^j}
\end{equation}

where $\mathcal F_s(n) := \{ \mathbf a \in \{ 1,\dots, n\}^s : \sum\_{k=1}^s a_k = n\}$. Moreover, Lemma 14 states that

\begin{equation}\label{eq:lemma14}
    \sum_{\mathbf a_j \in \mathcal F_{s_j}(n_j) } \left[ \frac{n_j^2}{\prod_{k=1}^{s_j} a_k^j}\right]^2 < C
\end{equation}

for a constant $C < 7$. Bringing these inequalities together, we get

\begin{equation}\label{eq:rntsbound}
    \mathbb E\[R(n,t,s)\]  \leq  \frac{2^t \sqrt{\log n}}{K}\left(\frac{UC}{m}\right)^s \sum_{\mathbf s \in \mathbf S} \prod_{j=1}^t \frac{1}{s_j!}\,.
\end{equation}

Noting that 

\begin{equation}
    \mathbb E\left[\sum_{s=1}^{n-t} \frac{\mathbb P(K_n = s+t \mid X_{1:n})}{\mathbb P(K_n = t\mid X_{1:n} )} \right] = \sum_{s=1}^{n-t} C(n,t,t+s) \mathbb E[R(n,t,s)\] \,,
\end{equation}


when Ascolani et al. combine $C(n,t,t+s)$ and $R(n,t,t+s)$, they obtain

<div class="scrollable-equation">
\begin{equation}\label{eq:subfact2}
  \mathbb E\left[\sum_{s=1}^{n-t} \frac{\mathbb P(K_n = s+t \mid X_{1:n})}{\mathbb P(K_n = t\mid X_{1:n} )} \right] \leq \frac{\left(\frac{2U}{m}\right)^tG\Gamma(t+\beta+1)\sqrt{\log n}}{K\varepsilon \log{\left(\frac{n}{1+\varepsilon} \right)}} \sum_{s=t}^{n-t} \frac{s \left(\frac{2UC}{m\rho}\right)^s\mathbb E[\alpha^{t+s-1}]}{(s+1)!}  
\end{equation}
</div>

and finally, using A3 on $\mathbb E[\alpha^{t+s-1}]$ with $\rho$ sufficiently large such that $2UC /m\rho < 1$, they obtain

<div class="scrollable-equation">
\begin{equation}\label{eq:shrink}
    \mathbb E\left[\sum_{s=1}^{n-t} \frac{\mathbb P(K_n = s+t \mid X_{1:n})}{\mathbb P(K_n = t\mid X_{1:n} )} \right] \leq \frac{\left(\frac{2U}{m}\right)^t \rho^{1-t} DG\Gamma(t+\beta+1)\sqrt{\log n}}{K\varepsilon \log{\left(\frac{n}{1+\varepsilon} \right)}} \sum_{s=t}^{n-t} \frac{s \left(\frac{2UC}{m\rho}\right)^s\mathbb \rho^{-s} \Gamma{(\nu + t+s)}}{(s+1)!}  
\end{equation}
</div>

which converges to $0$ as $n\to \infty$ since the factor before the sum shrinks to $0$ at a rate of $1/\sqrt{\log n}$ and the sum is finite as the exponential decay in $s$ beats the polynomial growth in $s$. Some additional Lemmas show that this result in (\ref{eq:shrink}) is sufficient for (\ref{eq:tozero}) and hence for consistency in (\ref{eq:post}).

#### Required Steps for our Extension
To extend the result of Ascolani et al. to Pitman-Yor processes, we now have to consider the partition distribution of the Pitman-Yor process, given in (\ref{eq:pympd}). Hence, we deal with the ratio 

<div class="scrollable-equation">
$$
\begin{equation}
   \frac{\mathbb P (K_n = s \mid X_{1:n} ) }{\mathbb P (K_n = t \mid X_{1:n} )}  =  \underbrace{\frac{  \int_{\mathbb R} \frac{(\alpha+d)_{s-1\uparrow d}}{(\alpha+1)_{n-1\uparrow 1}} \pi (\alpha) d\alpha}{\int_{\mathbb R} \frac{(\alpha+d)_{t-1\uparrow d}}{(\alpha+1)_{n-1\uparrow 1}} \pi (\alpha) d\alpha }}_{C'(n,t,s)} \cdot \underbrace{\frac{\sum_{\mathbf A\in\tau_{s}(n) }\left[ \prod_{j=1}^s (1-d)_{|A_j| - 1 \uparrow 1} \prod_{j=1}^s m(X_{A_j})\right] }{  \sum_{\mathbf B\in\tau_{t}(n) }\left[ \prod_{j=1}^t (1-d)_{|B_j| - 1 \uparrow 1} \prod_{j=1}^t m(X_{B_j})\right]}}_{R'(n,t,s)}\,.  
\end{equation}
$$
</div>

To derive a similar result to (\ref{eq:tozero}), we need to produce a bound on $R'(n,t,s)$ similar to (\ref{eq:rntsbound}). We can adapt Lemmas 3, 13 and 14 because they do not depend on the clustering process.  We also need to derive a similar bound to (\ref{eq:c1}) for $C'(n,t,s)$. 

### Results
We were able to extend the results of Ascolani et al. to Pitman-Yor $\mathrm{PY}(\alpha, d,Q_0)$ processes with a Uniform$(0,c)$ prior on $\alpha$ when $c$ and $d$ are fixed sufficiently close to $0$. Formally, we have that for the model

\begin{equation}\label{eq:pym2}
     X_i \mid \theta_i \ind k(\cdot \mid \theta_i),  \quad
    \theta_i \mid \widetilde{P} \iid \widetilde{P}, \quad 
    \widetilde{P} \sim \mathrm{PY}(\alpha, d, Q_0) \,, \quad \alpha\mid \mathrm{Uniform}(0,c)\,,
\end{equation}

the posterior $K_n$ on the number of clusters is consistent. To do this, we need to adapt the proof in Ascolani et al. in two ways.

#### New bound on R(n,t,s)
The bound on $R'(n,t,s)$ can have a factor that grows at most exponentially with $s$. To create this bound, we extend (\ref{eq:rntsbound}) and show that

<div class="scrollable-equation">
$$
\begin{equation}\label{eq:newrnts}
\sum_{\mathbf A_j \in\tau_{s_j}(n_j)} \frac{n_j}{\prod_{k=1}^{s_j} (|A_k^j| +1)}\cdot \frac{\prod_{k=1}^{s_j} (1-d)_{|A_k^j|-1\uparrow 1}}{(1-d)_{n_j - 1\uparrow 1}} \leq \frac{R^{s_j}}{s_j!} \sum_{\mathbf a_j\in \mathcal F_{s_j}(n_j)} \left[\frac{n_j}{s_j!\prod_{k=1}^s a_k^j}\right]^2 \leq \frac{(RC)^{s_j} }{s_j!}    
\end{equation}
$$
</div>

where $R=1/(1-d)$. Then (\ref{eq:rntsbound}) would hold with an additional $R$ term

$$
\begin{equation}\label{eq:newrnts2}
\mathbb E\left[R'(n,t,s)\right] \leq  \frac{2^t \sqrt{\log n}}{K}\left(\frac{UCR}{m}\right)^s \sum_{\mathbf s \in \mathbf S} \prod_{j=1}^t \frac{1}{s_j!}\,,
\end{equation}
$$

which indeed satisfies our requirements. In particular, the following is sufficient to show (\ref{eq:newrnts}).

#### Result 1:
 We have

$$
\begin{equation}
\sum_{\mathbf A\in \tau_s(n)}\frac{\prod_{j=1}^s (1-d)_{|A_k^j|-1\uparrow 1}}{ (1-d)_{n-1\uparrow1}} \leq \sum_{\mathbf a\in \mathcal F_s(n)} \frac{n}{s!\prod_{j=1}^s a_j}R^{s}
\end{equation}
$$

where $R := 1/(1-d)$ is a constant.

#### Proof:
We first take the same transform between summing over partitions $\tau_s(n)$ into vectors over $\mathcal F_{s}(n)$ like in Lemma 3 of Ascolani et al. to obtain

<div class="scrollable-equation">
$$
\begin{aligned}
    \sum_{\mathbf A\in \tau_s(n)} \frac{\prod_{k=1}^s (1-d)_{|A_k|-1\uparrow1}}{(1-d)_{(n-1)\uparrow 1}} &= \sum_{\mathbf A\in \tau_s(n)} \frac{\prod_{k=1}^s (1-d)\cdots(|A_k|-1-d) }{(1-d)\cdots(n-1-d)} \\
    &= \sum_{\mathbf a\in \mathcal F_{s}(n)} \frac{1}{s!}\cdot \frac{n!}{a_1!\cdots a_k!} \cdot \frac{\prod_{k=1}^s (1-d)\cdots(a_k-1-d) }{(1-d)\cdots(n-1-d)} \\
    &= \frac{1}{s!}  \sum_{\mathbf a\in \mathcal F_{s}(n)} \frac{n}{\prod_{k=1}^s a_k} \cdot \left[ \frac{(n-1)\cdots 1}{(n-1-d)\cdots (1-d)}\right]\prod_{k-1}^s \left[ \frac{(a_k-1-d)\cdots (1-d)}{(a_k-1)\cdots 1}\right]  \,. 
\end{aligned}
$$
</div>

It remains to show that 

$$
\begin{equation}
\left[ \frac{(n-1)\cdots 1}{(n-1-d)\cdots (1-d)}\right]\prod_{k-1}^s \left[ \frac{(a_k-1-d)\cdots (1-d)}{(a_k-1)\cdots 1}\right] < R^s
\end{equation}
$$

for $R= 1/(1-d)$.
Indeed, since the map $x\mapsto x/(x-\xi)$ for $\xi>0$ is decreasing, so for $n-s\geq 1$, we have  

$$
\begin{equation}
n-s\geq 1 \implies \frac{n-s}{n-s-d} \leq \frac{1}{1-d} \,,
\end{equation}
$$

and moreover,

$$
\begin{equation}
    1\leq \left[\frac{1}{1-d}\right]^s \left[ \frac{n-s-d}{n-s}\right]^s \leq R^s\left[1-\frac{d}{n-1}\right]\cdots \left[1-\frac{d}{n-s}\right]
\end{equation}
$$

If we arrange the components $a_k$ of $\mathbf a$ in increasing order, we have

$$
\begin{aligned}
    \prod_{k=1}^s \left[1-\frac{d}{a_k-1}\right]\cdots \left[1-\frac{d}{1}\right] \leq R^s\left[1-\frac{d}{n-1}\right]\cdots \left[1-\frac{d}{1}\right] 
\end{aligned}
$$

since $\sum_{j=1}^s (a_j-1) = n-s$. Rearranging gives us the desired inequality. 

Applying Result 1, we obtain

$$
\begin{equation}
\sum_{\mathbf A_j \in \tau_{s_j}(n_j)}\frac{\prod_{k=1}^{s_j} (1-d)_{|A_k^j|-1\uparrow 1}}{ (1-d)_{n_j-1\uparrow1}} \leq \sum_{\mathbf a_j\in \mathcal F_{s_j}(n_j)} \frac{n_j}{s_j!\prod_{j=1}^s a_k^j}R^{s_j} \,.
\end{equation}
$$

In particular, the right term is common with the product $n_j/\prod\_{k=1}^{s_j}(\|A_k^j\| + 1)$ in (\ref{eq:newrnts}), so we can use (\ref{eq:lemma14}) to bound our expression from above by an exponential on $s$, producing the required bound.

#### New bound on C(n,t,s)

Next, we consider $C'(n,t,s)$. In the case of the Uniform$(0,c)$ prior, we have 

\begin{equation}
    \mathbb E (\alpha^s) = \int_0^{\infty} \alpha^s \pi(\alpha) d\alpha = c^s \,.
\end{equation}

Our goal is to use the expansion

$$
\begin{multline}\label{eq:expand}
    (\alpha+d)_{t+s-1\uparrow d} = \underbrace{[M_0 \alpha^{t+s-1}d^0 + M_1 \alpha^{t+s-2}d^1 + \cdots + M_{s-2}\alpha^{t+1}d^{s-2}]}_{L_1(n,t,s)}\\+ \underbrace{[M_{s-1}\alpha^{t}d^{s-1}]}_{L_2(n,t,s)} + \underbrace{[M_{s}\alpha^{t-1}d^{s} + \cdots + M_{t+s-1} \alpha^0 d^{t+s-1}]}_{L_3(n,t,s)}
\end{multline}
$$

to reduce the ratio $C'(n,t,t+s)$ to a linear combination of $C(n,t,k)$ and apply known results like (\ref{eq:c1}) to create a suitable bound.
Note that the maximum coefficient in the expansion in (\ref{eq:expand}) is $M_{t+s-1} = (t+s-1)!$. There are three parts which we need to treat separately -- call these parts $L_1(n,t,s)$, $L_2(n,t,s)$ and $L_3(n,t,s)$ respectively. \

#### First Part:

We consider the first part of the expansion $L_1(n,t,s)$. We have for all $0<k<n$,

\begin{equation}
C(n,t,t+k) \leq  \frac{G\Gamma(t  +\beta + 1)2^k k }{\varepsilon\log\left(\frac{n}{1+\varepsilon}\right)}\mathbb E(\alpha^{t+k-1})
\end{equation}

by (\ref{eq:c1}). Since 

$$
\begin{aligned}
    L_1(n ,t, s) &:= M_0 \alpha^{t+s-1}d^0 + M_1 \alpha^{t+s-2}d^1 + \cdots + M_{s-2}\alpha^{t+1}d^{s-2} \\
    &\leq (t+s-1)! \sum_{j=0}^{s-2} \alpha^{t+s-1-j}d^j \,,
\end{aligned}
$$

we can produce a bound on the new ratio of integrals

$$
\begin{equation}\label{eq:s1}
\frac{  \int_{\mathbb R} \frac{L_1(n ,t,s)}{(\alpha+1)_{n-1\uparrow 1}} \pi (\alpha) d\alpha}{\int_{\mathbb R} \frac{(\alpha+d)_{t-1\uparrow d}}{(\alpha+1)_{n-1\uparrow 1}} \pi(\alpha)d\alpha}  \leq (t+s-1)! \sum_{j=0}^{s-2}d^j \cdot \frac{ \int_{\mathbb R} \frac{ \alpha^{t+s-1-j}}{(\alpha+1)_{n-1 \uparrow 1}} \pi(\alpha) d\alpha }{   \int_{\mathbb R} \frac{\alpha^t}{(\alpha + 1)_{n -1 \uparrow 1}}  \pi(\alpha) d\alpha } \,,
\end{equation}
$$

noting that the denominator is greater when $d$ is larger. 
Now, the choice of $G$, $\beta$ and $\varepsilon$ in (\ref{eq:c1}) depends on our choice of $c$ in the Uniform$(0,c)$ prior. Fortunately, they are constant factors that do not grow with the number of data points $n$ or the number of clusters $s$, so they do not affect our convergence results as $n\to\infty$. 

Next, we consider the sum in (\ref{eq:s1}). We have the bound

<div class="scrollable-equation">
$$
\begin{equation}
    \sum_{j=0}^{s-2}d^j \cdot \frac{ \int_{\mathbb R} \frac{ \alpha^{t+s-1-j}}{(\alpha+1)_{n-1 \uparrow 1}} \pi(\alpha) d\alpha }{   \int_{\mathbb R} \frac{\alpha^t}{(\alpha + 1)_{n -1 \uparrow 1}}  \pi(\alpha) d\alpha } \leq \frac{G\Gamma(t  +\beta + 1)}{\varepsilon\log\left(\frac{n}{1+\varepsilon}\right)} \sum_{j=0}^{s-2}d^j  2^{s-1-j} (s-1-j)\mathbb E(\alpha^{t+s-1-j}) \,.
\end{equation}
$$
</div>

Using $\mathbb E(\alpha^{t+s-1-j}) = c^{t+s-1-j}$, we have

<div class="scrollable-equation">
$$
\begin{aligned}
\sum_{j=0}^{s-2}d^j  2^{s-1-j} (s-1-j) c^{t+s-1-j} &=  2^{s-1}(s-1)c^{t+s-1} + 2^{s-2}(s-2)c^{t+s-2}d + \cdots + d^{s-3}2^2\cdot 2 c^2 + d^{s-2}2c \\
 &\leq c^{t+1}2^s s\left[\frac{d^{s-1} - c^{s-1}}{d-c}\right]
\end{aligned}
$$
</div>

so the ratio satisfies

<div class="scrollable-equation">
$$
\begin{equation}
    \frac{  \int_{\mathbb R} \frac{L_1(n ,t,s)}{(\alpha+1)_{n-1\uparrow 1}} \pi (\alpha) d\alpha}{\int_{\mathbb R} \frac{(\alpha+d)_{t-1\uparrow d}}{(\alpha+1)_{n-1\uparrow 1}} \pi(\alpha)d\alpha} \leq  \frac{G\Gamma(t  +\beta + 1) (t+s-1)! }{\varepsilon\log\left(\frac{n}{1+\varepsilon}\right)}\cdot \frac{c^{t+1}2^s s(d^{s-1} - c^{s-1})}{(d-c)} \,,
\end{equation}
$$
</div>

and hence,

<div class="scrollable-equation">
$$
\begin{equation}\label{eq:pc}
\sum_{s=1}^{n-t} \left[\frac{  \int_{\mathbb R} \frac{L_1(n ,t,s)}{(\alpha+1)_{n-1\uparrow 1}} \pi (\alpha) d\alpha}{\int_{\mathbb R} \frac{(\alpha+d)_{t-1\uparrow d}}{(\alpha+1)_{n-1\uparrow 1}} \pi(\alpha)d\alpha} \right]\mathbb E [ R'(n,t,s)] \leq 
\frac{2^tc^{t+1} \left(\frac{U}{m}\right)^t G \Gamma(t+\beta + 1) \sqrt{\log n}}{K(d-c)\varepsilon \log\left(\frac{n}{1+\varepsilon}\right)}
 \sum_{s=1}^{n-t}\frac{s\left(\frac{2UCR}{m}\right)^s (d^{s-1} - c^{s-1}) (t+s-1)!}{(s+1)!} 
\end{equation}
$$
</div>

when we factor out all of the common terms. For this to converge to $0$ like in (\ref{eq:shrink}), we require an exponentially decaying term in the sum. To get this term, we note that

$$
\begin{aligned}
    d^{s-1} - c^{s-1} \leq 2\max{(c,d)}^{s-1}
\end{aligned}
$$

so if we pick $c, d$ sufficiently small such that $\left(\frac{2UCR}{m}\right) \cdot d < 1$ and $\left(\frac{2UCR}{m}\right) \cdot c < 1$, then the sum would be finite as $n\to\infty$ since exponential decay beats polynomial growth. 
Note that the constant $R= 1/(1-d)$ depends on $d$, but this is not a problem because if $d\to 0$ is small, then we have $R \to 1$. So, we are free to let $d$ be very small if necessary, without fear that the $R$ term will blow up.

#### Second Part:

We can bound the $L_2(n, t, s) := M_{s-1}\alpha^{t}d^{s-1}$ term since $M_{s-1} \leq (t+s-1)!$ implies the inequality 

$$
\begin{equation}
    \frac{  \int_{\mathbb R} \frac{M_{s-1}\alpha^{t}d^{s-1}}{(\alpha+1)_{n-1\uparrow 1}} \pi (\alpha) d\alpha}{\int_{\mathbb R} \frac{(\alpha+d)_{t-1\uparrow d}}{(\alpha+1)_{n-1\uparrow 1}} \pi(\alpha)d\alpha} \leq d^{s-1}(t+s-1)!\,.
\end{equation}
$$

In particular, if $d$ is small enough such that $\left(\frac{2UCR}{m}\right) \cdot d < 1$, which was required in the first part, then

<div class="scrollable-equation">
$$
\begin{equation}
    \sum_{s=1}^{n-t} \left[\frac{  \int_{\mathbb R} \frac{M_{s-1}\alpha^{t}d^{s-1}}{(\alpha+1)_{n-1\uparrow 1}} \pi (\alpha) d\alpha}{\int_{\mathbb R} \frac{(\alpha+d)_{t-1\uparrow d}}{(\alpha+1)_{n-1\uparrow 1}} \pi(\alpha)d\alpha}\right]\mathbb E [ R'(n,t,s)] \leq 
\frac{2^t \left(\frac{U}{m}\right)^t G \Gamma(t+\beta + 1) \sqrt{\log n}}{K\varepsilon \log\left(\frac{n}{1+\varepsilon}\right)}
 \sum_{s=1}^{n-t}\frac{s\left(\frac{2UCR}{m}\right)^s d^s (t+s-1)!}{(s+1)!} 
\end{equation}
$$
</div>

so the sum would be finite as $n\to\infty$ since, again, the exponential decay of $\left(\frac{2UCR}{m}\cdot d\right)^s$ with $s$ beats the polynomial growth in $s$. \

#### Third Part:
Dealing with 

$$
\begin{equation}
    L_3(n,t,s) := M_{s}\alpha^{t-1}d^{s} + \cdots + M_{t+s-1} \alpha^0 d^{t+s-1}
\end{equation} 
$$

is different because we cannot set $d=0$ in the denominator of the ratio of integrals like in the previous cases. Instead, we use the observation that $L_3(n,t,s)$ contains a constant $t$ terms. We have that

$$
\begin{equation}
    (\alpha+d)_{t-1\uparrow d} = (\alpha+d)\cdots (\alpha+(t-1)d) \geq \alpha^{t-1} + \alpha^{t-2}d + \cdots + d^{t-1}
\end{equation}
$$

and also,

<div class="scrollable-equation">
$$
\begin{equation}
    M_{s}\alpha^{t-1}d^{s} + \cdots + M_{t+s-1} \alpha^0 d^{t+s-1} \leq t(t+s-1)! (\alpha^{t-1}d^s + \cdots d^{t+s-1} ) = t(t+s-1)!d^s(\alpha^{t-1} + \cdots + d^{t-1})
\end{equation}
$$
</div>

so the ratio of integrals is bounded by

<div class="scrollable-equation">
$$
\begin{equation}
    \frac{  \int_{\mathbb R} \frac{L_3(n ,t,s)}{(\alpha+1)_{n-1\uparrow 1}} \pi (\alpha) d\alpha}{\int_{\mathbb R} \frac{(\alpha+d)_{t-1\uparrow d}}{(\alpha+1)_{n-1\uparrow 1}} \pi(\alpha)d\alpha}  \leq \frac{  \int_{\mathbb R} \frac{t(t+s-1)!d^s(\alpha^{t-1} + \cdots + d^{t-1})}{(\alpha+1)_{n-1\uparrow 1}} \pi (\alpha) d\alpha}{\int_{\mathbb R} \frac{\alpha^{t-1}+\cdots + d^{t-1}}{(\alpha+1)_{n-1\uparrow 1}} \pi(\alpha)d\alpha} = t(t+s-1)! d^s
\end{equation}
$$
</div>

and since we chose $d$ such that $\left(\frac{2UCR}{m}\right) \cdot d < 1$ we have convergence in the sum.

Bringing it all together, we have the following result.

#### Result 2:
If $c, d$ are sufficiently small such that $\left(\frac{2UCR}{m}\right) \cdot d < 1$ and $\left(\frac{2UCR}{m}\right) \cdot c < 1$, then 

$$
\begin{equation}
    \sum_{s=1}^{n-t} C'(n,t,t+s) \mathbb E[R'(n,t,t+s)] \to 0 \text{ as $n\to\infty$ almost surely.}
\end{equation}
$$

#### Proof:
We have that $ C'(n,t,t+s) = L_1(n,t,s)+ L_2(n,t,s) + L_3(n,t,s)$ and by our discussion above,

<div class="scrollable-equation">
$$
\begin{multline}
    \sum_{s=1}^{n-t} C'(n,t,t+s) \mathbb E[R'(n,t,s)] = \sum_{s=1}^{n-t}\frac{  \int_{\mathbb R} \frac{L_1(n ,t,s)}{(\alpha+1)_{n-1\uparrow 1}} \pi (\alpha) d\alpha}{\int_{\mathbb R} \frac{(\alpha+d)_{t-1\uparrow d}}{(\alpha+1)_{n-1\uparrow 1}} \pi(\alpha)d\alpha} \mathbb E[R'(n,t,s)] \\ +\sum_{s=1}^{n-t} \frac{  \int_{\mathbb R} \frac{L_2(n ,t,s)}{(\alpha+1)_{n-1\uparrow 1}} \pi (\alpha) d\alpha}{\int_{\mathbb R} \frac{(\alpha+d)_{t-1\uparrow d}}{(\alpha+1)_{n-1\uparrow 1}} \pi(\alpha)d\alpha} \mathbb E[R'(n,t,s)]  + \sum_{s=1}^{n-t} \frac{  \int_{\mathbb R} \frac{L_3(n ,t,s)}{(\alpha+1)_{n-1\uparrow 1}} \pi (\alpha) d\alpha}{\int_{\mathbb R} \frac{(\alpha+d)_{t-1\uparrow d}}{(\alpha+1)_{n-1\uparrow 1}} \pi(\alpha)d\alpha} \mathbb E[R'(n,t,s)] \to 0
\end{multline}
$$
</div>

as $n\to \infty$ almost surely as each of the three parts converges to $0$ almost surely.

Following the rest of Ascolani et al., Result 2 is sufficient to show the consistency of the model in (\ref{eq:pym2}) when the specified assumptions and conditions are met. 



### Discussion

#### The Choice of Prior

The consistency results for the Dirichlet process in Ascolani et al. hold for a wide range of priors, including all distributions with bounded support that satisfy A1 and A2, like the Uniform$(0,c)$ distribution. Priors like the gamma distribution and the generalised gamma distribution can also attain consistency. These priors work because of the subfactorial moment condition A3. 

For the Pitman-Yor process, however, we were only able to show consistency for the specific case of the Uniform$(0,c)$ prior, where $c$ is small. We believe this is because the subfactorial moment condition does not provide enough decay to show the convergence of (\ref{eq:shrink}) when we replace the ratio of integrals $C(n,t,t+s)$ with $C'(n,t,t+s)$. 

When working with $C(n,t,t+s)$, the factorial growth rate of each moment $\Gamma(\nu+t+s)$ in A3 is slowed to a polynomial growth rate by the $1/(s+1)!$ term in the bound of $R(n,t,s)$, as seen in (\ref{eq:shrink}). However, $d$ is positive in the partition distribution of the Pitman-Yor process, so when we expand out $C'(n,t,t+s)$ into linear components of $C(n,t,t+s)$, we have an additional $(t+s-1)!$ factor. If we kept the subfactorial moment assumption on the prior, then $C'(n,t,t+s)$ would grow like the square of a factorial in $s$ instead. While we still have the $1/(s+1)!$ decay of $R'(n,t,s)$, the sum in (\ref{eq:subfact2}) would still grow like a factorial in $s$ instead of a polynomial in $s$, and blow up to infinity since factorial growth is stronger than exponential decay. Hence, the subfactorial moment assumption is too weak when using this strategy of proof with the Pitman-Yor process.

When using a Dirichlet process model, we are also free to choose a $\mathrm{Uniform}(0,c)$ prior for any $c$. That is because the factorial part of A3 dominates the exponential decay of $\rho^{-s}$, allowing us to choose $\rho$ arbitrarily large so that $1/\rho$ is arbitrarily small by compensating with a larger constant $D$ to rescale the upper bound if required. However, we are not allowed this flexibility for the Pitman-Yor model.
Since we can no longer have the moments of the prior grow like A3, we cannot choose an arbitrarily small $1/\rho$ either. Hence, only very small values of $c$ can be used in the Uniform$(0,c)$ prior to provide the exponential decay condition required in (\ref{eq:pc}). The same is true for the discount parameter $d$.

Altogether, it is certainly the case that the permissible choices of prior $\pi$ are much more limited with Pitman-Yor models.

#### Some Heuristic Discussion
Recall that the original problem identified in Miller and Harrison's work was that, heuristically, very small clusters are formed to accommodate outlier points. 

For a Dirichlet process $\mathrm{DP}(\alpha, Q_0)$, the expected number of predicted clusters is $\mathbb E(K_n) \approx \alpha \log(n)$, and the goal of Ascolani et al. was to show that the addition of a prior modifies the behaviour of the model so that a new cluster is generated with less probability as the number of data points rises, cancelling out this logarithmic growth. However, for a Pitman-Yor process $\mathrm{PY}(\alpha, d, Q_0)$, the expected number of clusters $\mathbb E(K_n)$ is roughly $\alpha n^{d}$. This grows faster than the Dirichlet process, and hence, if we are to have any chance of producing the same consistency result, then it seems intuitive that we must have a very small discount parameter $d$.

Moreover, introducing a Uniform$(0,c)$ prior on $\alpha$ where $c$ is small means we are imposing a strong belief that the rate at which new clusters should be formed is small. This can be seen in the partition distribution of the Pitman-Yor process and also makes sense considering the growth rate of the clusters.

While we were not able to provide a general condition required for consistency under the Pitman-Yor model, it seems reasonable to suggest from the heuristics that, as an alternative to the subfactorial moment assumption, we should instead consider families of priors that can be made to concentrate very strongly about $0$. 

### Insights

To place our work in the context of the literature, we note that the tendency of Pitman-Yor models to overestimate the number of clusters was empirically known before Miller and Harrison's proof (West et al., 1994, Lartillot and Phiippe, 2004, and others), so the results of Ascolani et al. were unexpected. 
Hence, we were interested to see if we could extend these results to Pitman-Yor processes with Miller and Harrison's paper in view, and indeed, we were able to show consistency under some conditions for Pitman-Yor processes using the same techniques as Ascolani et al. 

Our analysis of the $C'(n,t,s)$ and $R'(n,t,s)$ terms for the Pitman-Yor process model extends the study of the $C(n,t,s)$ and $R(n,t,s)$ terms in Ascolani et al. 
Extending $R'(n,t,s), $ was simple because it mainly encodes information about the data-generating process, so its dependence on the partition distribution is low. However, $C'(n,t,s)$ is highly dependent on the prior and the clustering process. As a result, we observed a large behavioural phase change upon setting $d>0$, raising complications that did not occur when $d=0$ for Dirichlet processes in Ascolani et al.

Interestingly, we can derive Miller and Harrison's result if we use a point mass $\delta_{\{a\}}$ as our prior to emulate a fixed concentration parameter $\alpha$. Since we need to consider the refinement of a partition into $t+1$ parts, the point mass prior reduces the ratio of integrals $C'(n,t,t+1)$ to a constant term, from which we can use the same approach as in the proof of Theorem 1 of Ascolani et al. to show the $R'(n,t,t+1)$ term does not disappear.  

However, a limitation of our work is that we only used the bounds derived from Ascolani et al. and focussed on reducing the Pitman-Yor partition distribution to cases that could be handled using the tools they developed for Dirichlet processes. This technique heavily restricted our choice of priors. It seems reasonable that, with effort, developing stronger bounds for Pitman-Yor processes compared to those in Ascolani et al. may allow us to show consistency for more priors.

### Future Work
There are several clear directions for future work. 

As part of our project, we showed a positive consistency result for the special case of a Uniform$(0,c)$ prior and Pitman-Yor process $\mathrm{PY}(\alpha, d,Q_0)$ where both $c$ and $d$ are very small. To extend this result, we can try to prove consistency for more types of priors. While we may not be able to form a general result, one possibility, as stated in the discussion, is to consider priors where the mass can be made to concentrate very closely around $0$ and explore if this can provide sufficient exponential decay in their moments. In the opposite direction, we could also research whether there exist conditions on the prior that allow us to derive a negative consistency result akin to Miller and Harrison.

Moreover, the work of Ascolani et al. assumed that the data points $X_1, X_2,\dots$ are independent. It would be interesting to explore whether we could still attain consistency when we instead apply the Dirichlet process to estimate the number of clusters in mixture models with a dependency structure, like a linear model with covariates.

### Conclusion
In this work, we studied the consistency results of Miller and Harrison and Ascolani et al. as described in the Aims. With Miller and Harrison, we saw how the probability of creating a new cluster under the Pitman-Yor model never disappears when the concentration and discount parameters are fixed, leading to the formation of spurious clusters. 
With Ascolani et al., we also saw that the addition of a prior is able to bring this probability down for a Dirichlet process as the number of data points increases.

We then extended the work of Ascolani et al. and showed consistency when using a Uniform$(0,c)$ prior for $\alpha$ in a Pitman-Yor model $\mathrm{PY}(\alpha, d, Q_0)$ with $c$ and $d$ small enough. However, we observed that extending to Pitman Yor models induced challenges that heavily restricted our choice of priors. In particular, the subfactorial moment assumption of Ascolani et al. was not sufficient to obtain consistency because the Pitman-Yor model tends to form more clusters than the Dirichlet process model.

Finally, we identified some avenues for future research, which include exploring the possible consistency of more general classes of priors and considering dependent mixtures.

### References

**[Aldous, 1985]** D.J. Aldous. Exchangeability and related topics. In: Hennequin, P.L. (eds) Ecole d'\'Et\'e de Probabilit\'es de Saint-Flour XIII - 1983. Lecture Notes in Mathematics, Vol 1117. Springer, Berlin, Heidelberg. 
`doi:10.1007/BFb0099421`

**[Ascolani et al., 2023]** F. Ascolani, A. Lijoi, G. Rebaudo and G. Zanella. Clustering consistency with Dirichlet process mixtures. Biometrika, 110(2), 551-558, 2023
`doi:10.1093/biomet/asac051`

**[Ferguson, 1973]** T. S. Ferguson. A Bayesian analysis of some nonparametric problems. The Annals of Statistics, pages 209-230, 1973.
`doi:10.1214/aos/1176342360`

**[Ishwaran and James, 2001]** H. Ishwaran and L. F. James. Gibbs Sampling Methods for Stick-Breaking Priors. Journal of the American Statistical Association, 96(453), 161–173, 2001 
`doi:10.1198/016214501750332758`

**[Lartillot and Phillipe, 2004]** N. Lartillot and H. Philippe. A Bayesian mixture model for across-site heterogeneities in the amino-acid replacement process. Molecular Biology and Evolution}, 21(6):1095-1109, 2004.
`doi:10.1093/molbev/msh112`


**[Miller and Harrison, 2014]** J. W. Miller and 
M. T. Harrison. Inconsistency of Pitman-Yor process mixtures for the number of components. Journal of Machine Learning, Res. 15, 3333–3370, 2014
`doi:10.48550/arXiv.1309.0024`

**[Onogi et al., 2011]** A. Onogi, M. Nurimoto, and M. Morita. Characterization of a Bayesian genetic clustering algorithm based on a Dirichlet process prior and comparison among Bayesian clustering methods. BMC Bioinformatics, 12(1):263, 2011.
`doi:10.1186/1471-2105-12-263`

**[Pella and Masuda, 2006]** J. Pella and M. Masuda. The Gibbs and split-merge sampler for population mixture analysis from genetic data with incomplete baselines. Canadian Journal of Fisheries and Aquatic Sciences, 63(3):576-596, 2006
`doi:10.1139/f05-224`

**[Pitman, 1994]** J. Pitman. Exchangeable and Partially Exchangeable Random Partitions, Probability Theory and Related Fields 102, 145–158, 1995.
`doi:10.1007/BF01213386`

**[Pitman and Yor, 1997]**  J. Pitman and M. Yor. The two-parameter Poisson-Dirichlet distribution derived from a stable subordinator. The Annals of Probability, 25(2):855-900, 1997
`doi:10.1214/aop/1024404422`

**[Pitman, 2002]** J. Pitman. Combinatorial Stochastic Processes, Technical Report, Technical Report 621, Department of Statistics, UC Berkeley, 2002.
`url: http://works.bepress.com/jim_pitman/`

**[West et al., 1994]** M. West, P. Muller, and M. D. Escobar. Hierarchical priors and mixture models, with application in regression and density estimation. In P. Freeman and A. F. Smith, editors,  Aspects of Uncertainty: A Tribute to D.V. Lindley, pages 363-386. Wiley, 1994.