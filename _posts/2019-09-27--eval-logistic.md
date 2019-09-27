---
layout: post
title: How to evaluate the value of an added independent variable in binary logistic regression models?  
subtitle: About AUC, peudo-R2, likelihood ratio test
tags: [stats]
---

We often predict the probability of occurence of binary outcomes: e.g. cases or controls using logistic regressions.

Specifically, we are building Polygenic risk score (PRS) that partially capture an individual's genetic risk for a trait/disease. 

Other than PRS, we usually have the individual's age and genotype PCs that could also help the prediction.

Then, a very natural question to ask is how well this PRS help us predicting the outcome?

There are many ways to evaluate this, but seems many can be simplified to one of two types: 

- a statistic measure how well you predict the outcome, usually the higher the better but there's no fixed cutoff to distinguish good or bad model
e.g. AUC, pseudo-R2

## AUC:

area under ROC curve:

each threshold is a point on the curve:

x axis: FPR = FP/FP+TN

y axis: TPR = TP/TP+FN

interpret: AUC is the probability of correct ranking of a random “positive”-“negative” pair.

## psedo-R2

### Maddala and Cox and Snell:

$$R^{2}_{MCS}=1-\frac{}{L(Null)}{L(Full)})^{2/N}$$

full: full model

null: intercept-only model

interpret: geometric mean square improvement

### Nagelkerke

since $$R^{2}_{MCS}$$ can exceed 1, rescale this to (0,1)

$$R^{2}_{NK} = \frac{1-\frac{}{L(Null)}{L(Full)})^{2/N}$$ }{ 1- L(Null)^{2/N}}$$

### McFadden

$$R^{2}_{MF}= 1 - \frac{LL(Full)}{LL(Null)}$$

1 minus ratio of full-model log-likelihood to intercept-only log-likelihood




- a goodness-of-fit test: tells us if it is beneficial to add parameters to our model, or if we should stick with our simpler model.
e.g. likelihood ratio test (LRT); Hosmer-Lemeshow test (HLT)

### LRT

In order to do this, we find the log-likelihoods of each model and plug them into the formula -2 * [loglikelihood(small)-loglikelihood(full)]. 
Our test statistic follows a chi-squared distribution with degrees of freedom equal to the difference in the number of free parameters between the complex model and the nested model. 
With this information, we may calculate the p-value, and if it is less than our significance level, we reject the null hypothesis.

### HLT

We group cases into deciles based on the predicted probability of each, then acess the degree to which the observed frequencies match the expected frequencies using a chi-square goodness-of-fit test, 
and a non-significant test result suggest a well-fitting model


references:

<http://www.glmj.org/archives/articles/Smith_v39n2.pdf>

<http://api.rpubs.com/tomanderson_34/lrt>

<https://statisticalhorizons.com/wp-content/uploads/GOFForLogisticRegression-Paper.pdf>

<https://www.alexejgossmann.com/auc/>
