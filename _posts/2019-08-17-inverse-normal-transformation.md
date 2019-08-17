---
layout: post
title: inverse normal transformation: why and how?
tags: [data transformation, statistics]
---

In statistical genetics field, we often observe non-normal distributed data. Also, we use linear models and parametric tests a lot. 
Linear models and parametric tests often assume residuals are normaly distributed. Then how do we respond to non-normality? 

Inverse-normal transformation is seen a lot in literature. There are also other transformation techniques and non-parametric tests available. 
For this post, we just focus on inverse-normal transformation.

# what is inverse-normal transformation?

There are several types of inverse-normal transformations: non-rank based vs rank based.
We talk about the commonly used rank based one.
The idea is simple: first converting a variable to ranks, and then back transform of sample quantile/fractional rank to approximate the expected normal scores.

$$ Y = \phi^{-1}(\frac{r-c}{N-2c+1})$$

Blom recommend: $$c=\frac{3}{8}$

Rankit：c = 0.5 (also used a lot)

# how to do inverse-normal transformation?

Do a normal quantile transform to phenotypes (replacing the rth biggest of N observations with the (r − 0.5)/Nth quantile of the standard normal distribution) before applying our methods. 

R code to do this (thanks to <https://www.biostars.org/p/80597/>):

```
qnorm((rank(x,na.last="keep")-0.5)/sum(!is.na(x)))
```

# does it have guaranteed type I error rate? 

The tests of group differences in location (e.g., mean) of inverse-normal transformed variables doesn't ganrantee type I error rate from this papaer
<https://link.springer.com/article/10.1007%2Fs10519-009-9281-0>. 
Since inverse-normal transformation only make sure the marginal distribution is normal, it doesn't make the residuals normal by definition.
But in reality, since "However, in this context, where effect sizes are expected to be generally rather small, normality of phenotype and normality of residuals are somewhat similar assumptions", 
it'll probably work in many genetics studies (taken from <https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0030114>). 
That could also be the reason that it is widely used in the field.




