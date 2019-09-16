---
layout: post
title: Linear model regularization
tags: [statistics]
---



# Linear Model Selection and Regularization

$$Y = \beta_{0} + \beta_{1} X_{1} + ... + \beta_{p} X_{p} + \epsilon$$

linear models have advantages in inference and works well in real world

how to improve on linear models?

p > N: the variance is infinite (more columns than rows), so no longer unique least square solution

why?
Think about augmented matrix. To get solution, we transform it into row-reduced echelon form.
For example, we end up with a 2x3 matrix, there's no unique solution for x1, x2, x3 to meet 2 equations.
We will have no solutions whenever we end up with one or more rows of all 0s except in the last column as we reduce the augmented matrix.

There are 3 important classes of method to improve on linear models.

# 2. Three classes of methods to improve linear models

# 2.1 subset selection

This approach involves identifying a subset of the p
predictors that we believe to be related to the response. We then fit
a model using least squares on the reduced set of variables

- best subset: $$2^{p}$$ computation heavy
- forward stepwise: $$1+ \sum{k=0}^{p} p-k = 1+ \frac{p(p+1)}{2}$$ may miss best solution
- backward stepwise: $$1+ \frac{p(p+1)}{2}$$ not work for n>p , may miss best solution

# 2.2 shrinkage

This approach involves fitting a model involving all p predictors.
However, the estimated coefficients are shrunken towards zero
relative to the least squares estimates. This shrinkage (also known as
regularization) has the effect of reducing variance. Depending on what
type of shrinkage is performed, some of the coefficients may be estimated
to be exactly zero. Hence, shrinkage methods can also perform
variable selection.

# 2.2.1 Ridge

The ridge solutions are not equivariant under scaling of the inputs, and
so one normally standardizes the inputs before solving

$$x_{ij}$$$$ replaced by $$x_{ij}-\bar{x_{j}}$$$$

minimize sum of residual sum of squares and a penalty term

$$\hat{\beta} = argmin (y-X\beta)^{T}(y-x\beta)+\lambda ||\beta||_{2}^2$$

In matrix form,

$$RSS(\lambda)=(y-X\beta)^{T}(y-X\beta)+\lambda \beta^{T}\beta$$

$$$$\hat{\beta}=(X^{T}X+\lambda I)^{-1} X^{T}y$$

This: add positive constant to diagnonal of $$X^{T}x$$ before inversion: make it nonsingular

intuition: since sample size is small, we want the model to be less sensitive to training sample, to do that, we make the predictors (x) has less impact on the outcome (y). To do that, we make $$\beta$$ smaller for each predictor, this way we can reduce the variance due to too many predictors.

It has a bayesian interpretation.

assume X: fixed

OLS:
$$y|x,\beta \sim N(X\beta, \sigma^2 I)$$

we impose a prior distribution on $$\beta_{j}$$, independent of each other 
$$\beta_{j} \sim N(0,\tau^2)$$

$$\beta \sim N(0,\tau^2 I)$$

we assume x is fixed, so $$p(X,\beta)=p(\beta)p(X)$$

$$p(\beta| y, X)=\frac{p(\beta,y,X)}{p(y,X)}=\frac{p(y|X,\beta)p(X,\beta)}{p(y,X)} \propto p(y|X,\beta)p(\beta)$$

$$p(y|X,\beta)p(\beta) = exp[-\frac{1}{2} (y-x\beta)^{T}\frac{1}{\sigma^2} (y-x\beta) ]exp[-\frac{1}{2} (\beta-0)^[T]\frac{1}{\tau^2} (\beta-0)]= exp(\frac{1}{2\sigma^2} (y-x\beta)^{T}(y-x\beta)-\frac{1}{2\tau^2}||\beta||^{2})$$


to maximize
$$exp(\frac{1}{2\sigma^2} (y-x\beta)^{T}(y-x\beta)-\frac{1}{2\tau^2}||\beta||^{2})$$

is the same as minimize

$$\frac{1}{2\sigma^2} (y-x\beta)^{T}(y-x\beta)+\frac{1}{2\tau^2}||\beta||^{2}$$

same as minimize

$$(y-x\beta)^{T}(y-x\beta)+\frac{\sigma^2}{\tau^2}||\beta||^{2}$$

$$\lambda = \frac{\sigma^2}{\tau^2}$$

## implement ridge



# 2.2.2 Lasso

$$||\beta||_{1}=\sum_{j=1}^{p}|\beta_{j}|$$

need predictors on same scale

minimize:

$$\beta^{lasso}=argmin (y-x\beta)^{T}(y-x\beta)+\lambda ||\beta||_{1}$$

bayesian lasso:

posterior distribution of $$\beta$$

prior: independent and identical laplace (double-exp) with mean 0 and scale $$\tau$$

laplace: $$p(x|\mu,b)=\frac{1}{2b}exp(-\frac{|x-\mu|}{b})$$

here, $$\beta \sim Laplace(0,\tau)$$

$$p(\beta_{j}|\tau) = \frac{1}{2\tau}exp(-\frac{|\beta_{j}|}{\tau})$$

$$p(\beta|\tau)=\prod_{j=1}^{p} p(\beta_{j}|\tau) = (\frac{1}{2\tau})^{p} exp(\frac{- \sum_{j=1}^{p} |\beta_{j}| }{\tau})$$

$$y_{i} \sim N(x\beta,\sigma^2)$$

here $$y=(y_{1},...,y_{n})$$

$$f(y|x,\beta,\sigma^2) \propto (\sigma^2)^{-n/2} exp(- \frac{1}{2\sigma^2}(y-x\beta)^{T}(y-x\beta))$$

to maximize
$$f(\beta|y,x) \propto p(\beta)p(y|x,\beta) = (\frac{1}{2\tau})^{p} exp(\frac{- \sum_{j=1}^{p} |\beta_{j}| }{\tau}) (\sigma^2)^{-n/2} exp(- \frac{1}{2\sigma^2}(y-x\beta)^{T}(y-x\beta))$$

log it
$$\frac{- \sum_{j=1}^{p} |\beta_{j}| }{\tau}- \frac{1}{2\sigma^2}(y-x\beta)^{T}(y-x\beta)$$

same as minimize
$$\frac{ \sum_{j=1}^{p} |\beta_{j}| }{\tau} + \frac{1}{2\sigma^2}(y-x\beta)^{T}(y-x\beta)$$

this is Lasso

# implement lasso from scratch: given lambda, find beta

We seek to obtain a sparse set of weights by minimizing the LASSO cost function

$$\sum_{i=1}^{n}(y_{i}-\hat{y_{i}})^{2}+ \sum_{j=1}^{p} \lambda |\beta_{j}|$$ 

The absolute value sign makes the cost function non-differentiable, so simple gradient descent is not viable (you would need to implement a method called subgradient descent). 

Instead, we will use **coordinate descent**: at each iteration, we will fix all weights but $$\beta_{i}$$ and find the value of $$\beta_{i}$$ that minimizes the objective. 

That is, we look for

# loss function

argmin 
$$L= \sum_{i=1}^{n}(y_{i}-\hat{y_{i}})^{2}+ \sum_{j=1}^{p} \lambda |\beta_{j}|$$

To get this minimum: we can take derivative and make it 0:

$$\frac{dL}{d\beta_{j}} = 0$$

but the derivative of the absolute is undefined at $$\beta_{j}=0$$

## loss part 1: OLS

we first look at the OLS part

$$\frac{d \sum_{i=1}^{n} [ y_{i} -\sum_{j=1}^{p} \beta_{j}x_{j}]^{2}}{d\beta_{k}} = 2[y_{i} -\sum_{j=1}^{p} \beta_{j \neq k }x_{j}-\beta_{k}x_{k}](-x_{k}) = -2[y-\sum_{j=1}^{p} \beta_{j \neq k }x_{j}]x_{k} + \beta_{k}x_{k}^{2}$$

we write $$p_{k}=\sum_{i=1}^{n}2[y_{i}-\sum_{j=1}^{p} \beta_{j \neq k }x_{j,i}]x_{k,i}$$

$$x_{k}^{2}=z_{k}$$

so the above equation can be written as
$$\frac{d \sum_{i=1}^{n} [ y_{i} -\sum_{j=1}^{p} \beta_{j}x_{j}]^{2}}{d\beta_{k}} = -p_{k}+\beta_{k}z_{k}$$


## loss part 2: absolute lambda

if $$\beta_{j}>0$$
$$\frac{d|\beta_{j}|\lambda }{d \beta_{j}}= \lambda$$

if $$\beta_{j}<0$$
$$\frac{d|\beta_{j}|\lambda }{d \beta_{j}}= -\lambda$$

if $$\beta_{j}=0$$
$$\frac{d|\beta_{j}|\lambda }{d \beta_{j}}=[ -\lambda, \lambda]$$


## combine 2 parts

we know the loss function is the sum of OLS part and penalty part

Use subdifferential rule:
$$\partial (f+g) = \partial (f) + \partial(g)$$

$$\frac{dL}{d\beta_{k}} = \frac{d OLS}{d\beta_{k}} + \frac{d|\beta_{j}|\lambda }{d \beta_{j}} = 0$$ 

this became

if $$\beta_{k}>0$$

$$-p_{k} + \beta_{k} z_{k} + \lambda =0$$

if $$\beta_{k}<0$$

$$-p_{k} + \beta_{k} z_{k} - \lambda =0$$

if $$\beta_{k}=0$$

$$-p_{k} + \beta_{k} z_{k} - [\lambda, \lambda] =0$$

we must ensure the closed interval contails 0 so that $$\beta_{k}=0$$ is a global min.

this means: $$0 \in [-p_{k} + \beta_{k} z_{k} - \lambda,-p_{k} + \beta_{k} z_{k} + \lambda]=[-p_{k} - \lambda,-p_{k} + \lambda]$$

so that 
if $$p_{k} > \lambda$$,
$$\beta_{k} = \frac{p_{k}-\lambda}{z_{k}}$$

if $$p_{k} < -\lambda$$,
$$\beta_{k} = \frac{p_{k}+\lambda}{z_{k}}$$

if $$-\lambda < p_{k} < \lambda$$, 
$$\beta_{k}=0$$

this is the soft threshold function $$\frac{1}{z_{k} S(p_{k},\lambda)}$$

# Coordinate Descent algo:

1. pick a coordinate k
2. compute $$p_{k}= \sum_{i=1}^{n}2[y_{i}-\sum_{j=1}^{p} \beta_{j \neq k }x_{j,i}]x_{k,i}$$
3. compute $$z_{k}=\sum_{i=1}^{n} x_{k,i}^{2}$$
4. set $$\beta_{k}=\frac{1}{z_{k} S(p_{k}, \lambda)}$$








# 2.3 dimension reduction

This approach involves projecting the p predictors
into a M-dimensional subspace, where M <p. This is achieved
by computing M different linear combinations, or projections, of the
variables. Then these M projections are used as predictors to fit a
linear regression model by least squares.


# 3. choose best model

the model containing all of the predictors will always have the smallest RSS and the
largest R2, since these quantities are related to the training error. Instead,
we wish to choose a model with a low test error


#3.1 adjust R2

We can indirectly estimate test error by making an adjustment to the
training error to account for the bias due to overfitting.

#3.2 cross validation

We can directly estimate the test error, using either a validation set
approach or a cross-validation approach, as discussed in Chapter 5.

