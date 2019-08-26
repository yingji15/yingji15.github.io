---
layout: post
title: generate lots of string combinations at once
subtitle: GNU parallel
tags: [utility]
---

In our work, we often need to generate lots of similar commands at once. 
There are lots of ways to do this, I just want to write about the one I used the most often: Parallel




# basic syntax

## The file name: {}

## remove extension: {.}

parallel echo {.} ::: A/B.C

Output: A/B
  
## removes the path: {/} 

parallel echo {/} ::: A/B.C

Output: B.C

## keep only path: {//} 

parallel echo {//} ::: A/B.C

Output: A

## remove path and extension: The replacement string {/.} 

parallel echo {/.} ::: A/B.C

Output:B


## To indicate that everything that follows should be read in from the command line: :::
e.g. "parallel gzip ::: *" means to gzip all files in the current working directory, while "parallel gzip *" wont work. You need to include ":::".



# my examples

## 1. 

For example, I need to generate simulations using Rscript, and then run it together using 10 threads.

With multiple input sources the argument from the individual input sources can be accessed with {number}:

We save commands to a file named cmd.

And we want all jobs in file to run in Parallel. If more jobs exist than jobs allowed, a queue is formed and maintained by Parallel until all jobs have run.

```
parallel echo Rscript test.r --s {1} --b1 {2} --b2 {3} --h {4} --cor {5} ::: 20 40 60 80 100 ::: 20 40 60 80 100 ::: 20 40 60 80 100 ::: 0.05 0.1 0.15 0.2 ::: 0.2 0.4 0.6 0.8 1 > cmd

less cmd | parallel -j 10
```


## 2. print first field to new file

parallel awk \''{print $1}'\' {} \> {}.xxx ::: random10*



## helpful posts on useful examples:

<https://gist.github.com/Brainiarc7/7af2ab5e88ef238da2d9f36b4be203c0>

<https://github.com/LangilleLab/microbiome_helper/wiki/Quick-Introduction-to-GNU-Parallel>
