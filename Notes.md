# Changepoint Detection — Notes

Accumulated theory/knowledge, kept separate from `Intro-work.ipynb` so that notebook stays focused on computation. Source material: https://www.lancaster.ac.uk/~romano/teaching/2425MATH337/5_real_data.html

# MLE - Quick Recap
The MLE is a method of estimating parameters of an assumed probability distribution, Given some observed data so under the statistical model and under the observed data the paramaters are most optimal. This point which maximises probability is known as the point in the parameter space as the MLE and is a means of statistical inference.

## The Kernel trick 
**Simply Put** it is a concept taking for functional analysis and allows us to take the data into a higher dimension and  function apply the inner product to out data in order to create a gram matrix which allows us to find similarites between our data points (which can be a higher dimensional space) 
The kernel trick essentially skips this entire part which can be incredibly computationally expensive and allows for you to take the kernel, this if complying with mercers theorem, that the kernel being PSD and all values being GOE 0., I will complete the section on Mercers theorem once I understand more
## Parametric vs. non-parametric methods

**Parametric methods** in statistics assume that the underlying population follows a known probability distribution — such as linear regression, Pearson correlation, t-tests, and ANOVA — as they are built off a likelihood ratio framework. Parametric models require: a large sample size, a distribution assumption, homogeneity of variance, and independence.

**Non-parametric methods** are statistical techniques that do not specify a probability distribution for the data, and instead work by estimating the population mean or variance directly.

## Segmentation accuracy metrics

Source: https://arxiv.org/pdf/1801.00718

Possible metrics for measuring how accurate a segmentation is: the **F1 metric**, the **Rand index**, and the **Hausdorff distance**.

---
### The difference between different kernels part 1
**informal idea restricted to variables whos decay exponentially or faster and blows up when its polynomially or slower need to look at Bochner;s theorem**

**Idea**: We are trying to understand the differences between the kernels, that being our linear, polynomial and our gaussian and why the gaussian is better

**Definitions**:
We must first define what characteristic is:
A kernel $K$ is said to be *Characteristic* if the mapping it induces from probability distributibution is injective or 
$$\mu_P = \mathbb{E}_{X \sim P}[K(X, \cdot)]$$
This maps P to a single point in the RKHS and the lap loses no distributional information. So that means that 
$$\mu_P &= \mu_Q$$ for 2 distributions $P$ and $Q$ therefore $P = Q$ where no two distinct distributions collapse on same embedding.

This idea connects to the idea of moment-expansion where the sequence of moments of our distrbution e.g. mean variance skewness etc we can find it. The kerne must encode all of these moments not just be a subset.

When we take our Linear Kernel it only captures the first moment, therefore not charactersitc since 2 distributions with the same mean collapse to the same embedding.


**Definition.** For a kernel $K$ and distribution $P$, the *kernel mean embedding* is
$$\mu_P(\cdot) = \mathbb{E}_{X\sim P}[K(X,\cdot)].$$
$K$ is **characteristic** if $P \mapsto \mu_P$ is injective, i.e. $\mu_P = \mu_Q \iff P=Q$.

A polynomial kernel (degree d), this captures moments up to order d, so therefore not charcteristic for any finite d, since any distributions agreeing for the first d moments will practically be indistingusiable.

#### Polynomial kernel (degree $d$) — not characteristic for any finite $d$

$K(x,y) = (xy+1)^d = \sum_{k=0}^d \binom{d}{k} x^k y^k$ (binomial theorem). Taking expectations,
$$\mu_P(y) = \sum_{k=0}^d \binom{d}{k}\, m_k(P)\, y^k, \qquad m_k(P) := \mathbb{E}_{X\sim P}[X^k].$$
So $\mu_P$ is exactly the polynomial whose coefficients are the moments of $P$ up to order $d$:
$$\mu_P = \mu_Q \iff m_k(P)=m_k(Q) \text{ for } k=0,\dots,d.$$


Our Guassian Kernel however has a taylor expansion which encodes every finite moment, due to its taylor expansion being infinite. This is why its characteristic over a very broad class of distributions and is the backbone of mehtods like MMD which uses this property.

#### Gaussian (RBF) kernel — characteristic

$$K(x,y) = \exp\!\left(-\frac{\|x-y\|^2}{2\sigma^2}\right) = e^{-\|x\|^2/2\sigma^2}\, e^{-\|y\|^2/2\sigma^2}\, e^{x^\top y/\sigma^2}.$$
Taylor-expanding the last factor,
$$e^{x^\top y/\sigma^2} = \sum_{k=0}^\infty \frac{(x^\top y)^k}{k!\,\sigma^{2k}},$$
so — unlike the polynomial kernel, which truncates at degree $d$ — the Gaussian kernel sums monomial kernels of **every** order at once. Informally, $\mu_P$ encodes a weighted moment of every order, so nothing is thrown away.



### F1 metric

**Idea:** count how many breakpoints were found "close enough" to a true breakpoint (within a tolerance $M$), then turn that count into a precision/recall trade-off.

Let $\mathcal{T} = \{t_1^*, \dots, t_n^*\}$ be the true breakpoints and $\hat{\mathcal{T}} = \{\hat t_1, \dots, \hat t_m\}$ be the estimated breakpoints. Define the match count

$$T = \left|\{\hat t \in \hat{\mathcal T} : \exists\, t^* \in \mathcal T,\ |\hat t - t^*| < M \}\right|$$

i.e. the number of estimated breakpoints that lie within $M$ of some true breakpoint.

| Quantity | Formula | Meaning |
|---|---|---|
| Precision | $P = \dfrac{T}{m}$ | fraction of estimated breakpoints that are correct |
| Recall | $R = \dfrac{T}{n}$ | fraction of true breakpoints that were found |
| F1 score | $F_1 = \dfrac{2PR}{P+R}$ | harmonic mean of precision and recall |

$F_1 \in [0,1]$, with $F_1 = 1$ only when every estimated breakpoint matches a true one ($P=1$) and every true breakpoint is recovered ($R=1$).

---

### Rand index

**Idea:** treat segmentation as clustering the $n$ points into segments, then measure agreement between the true and estimated segmentations over all $\binom{n}{2}$ pairs of points.

$$RI = \frac{a + b}{a + b + c + d} = \frac{a+b}{\binom{n}{2}}$$

| Symbol | Meaning |
|---|---|
| $a$ | pairs of points placed in the *same* segment in both segmentations |
| $b$ | pairs of points placed in *different* segments in both segmentations |
| $\binom{n}{2}$ | total number of pairs of the $n$ points ($=a+b+c+d$) |

So to find this index we add together all points that are similar, and all points that are different, and divide by all possible pairs of points. Take the example of the stock market — comparing a true vs. estimated market-regime segmentation:

- $A$ — agree, same segment (both agree on market regime, e.g. volatile–volatile)
- $B$ — agree, different segment (both agree the regimes differ, e.g. volatile–calm)
- $C$ — disagree, false split (estimate says volatile–calm when in reality it's volatile–volatile)
- $D$ — disagree, false merge (estimate says volatile–volatile when in reality it's volatile–calm)

---

### Hausdorff distance

**Idea:** This comes from metric spaces and can be described as being the biggest value, with points with their closest neighbour.

### Power curve

**Idea:** The Power Chart 

The Power chart shows how a change in variable will have a subsequent change in power (Detection rate) and can be used to find the accuracy of a method