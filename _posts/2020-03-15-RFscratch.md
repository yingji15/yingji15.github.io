---
layout: post
title: Random Forest from scratch
subtitle: "top down" approach
tags: [RF]
---

# RF from scratch

Although there are lots of "ML from scratch" online, I haven't seen many good RF ones.

In Lecture 7, Jeremy gave a good online tutorial on RF from scratch.(http://course18.fast.ai/lessonsml1/lesson7.html)

Jeremy introduced a concept new to me: "top down" programming: assume you have everything you need, how would you proceed

# given tree, how to get forest?

For this example: suppose we have a tree API, how to make a forest?

We need to sample x and y, and make predictions based on different trees, at last, prediction based on average of trees


# given that we know how to get 1 feature's best split, how to solve all features

# given feature, how to get best split?

So we delay "real work" as long as possible, and finish it before we realize! (I don't think it's that simple though...But this can help get myself started.)


Think about the process:

1. start from all samples (score: Inf, since this is the worst model)

2. get trees: loop through each col, loop through all split points:

3. get a prediction for each tree	 

4. avg predictions for forest result


```python
class TreeEnsemble():
# top down approach to write
# delay "real work" as long as possible
# assume we have everything to start with: imagine have tree API when writing RF
    def __init__(self, x, y, n_trees, sample_sz, min_leaf=5):
    	# write this first
        np.random.seed(42)
        self.x,self.y,self.sample_sz,self.min_leaf = x,y,sample_sz,min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]

    def create_tree(self):
    	# how to we predict tree?
    	# sample with replacement, (permutation) then subsample the first sample sz items
        idxs = np.random.permutation(len(self.y))[:self.sample_sz] # this line is used everywhere
        return DecisionTree(self.x.iloc[idxs], self.y[idxs], 
                    idxs=np.array(range(self.sample_sz)), min_leaf=self.min_leaf)
        
    def predict(self, x):
    	# write this 2nd
        return np.mean([t.predict(x) for t in self.trees], axis=0)

def std_agg(cnt, s1, s2): return math.sqrt((s2/cnt) - (s1/cnt)**2)


class DecisionTree():
	# plain tree: don't do random sample, just deal with data at hand
    # pass in random sample of x, sample of y, idxs(of data we want to use, since recursive start with entire random sample)
    def __init__(self, x, y, idxs, min_leaf=5):
        self.x,self.y,self.idxs,self.min_leaf = x,y,idxs,min_leaf
        self.n,self.c = len(idxs), x.shape[1] # n: observations, c: columns
        self.val = np.mean(y[idxs]) #  mean of y for those indexes (top: all data)
        self.score = float('inf')
        self.find_varsplit()
        
    def find_varsplit(self):# find variable to split
        for i in range(self.c): self.find_better_split(i) # go through all columns
        if self.score == float('inf'): return
        x = self.split_col
        lhs = np.nonzero(x<=self.split)[0] # convert a vector of 0/1 to only those 1 indexes
        rhs = np.nonzero(x>self.split)[0]
        self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs]) # build a tree on this half
        self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs])

    def find_better_split(self, var_idx):
    	# this is a slower implementation: o(n^2)
    	x,y = self.x.values[self.idxs,var_idx], self.y[self.idxs]


    	for i in range(self.n): # o(n)
        	lhs = x<=x[i] #o(n)
        	rhs = x>x[i]
        	if rhs.sum()<self.min_leaf or lhs.sum()<self.min_leaf: continue
        	lhs_std = y[lhs].std()
        	rhs_std = y[rhs].std()
        	curr_score = lhs_std*lhs.sum() + rhs_std*rhs.sum()
        	if curr_score<self.score: 
            	self.var_idx,self.score,self.split = var_idx,curr_score,x[i]

    def find_better_split(self, var_idx): # find a better split for this column
    # this way: o(n), we start everything at right side, then move 1 to left or not
    	# better: 2 groups have low sum of std dev
        x,y = self.x.values[self.idxs,var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y,sort_x = y[sort_idx], x[sort_idx]
        rhs_cnt,rhs_sum,rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
        lhs_cnt,lhs_sum,lhs_sum2 = 0,0.,0.

        for i in range(0,self.n-self.min_leaf):
            xi,yi = sort_x[i],sort_y[i]
            lhs_cnt += 1; rhs_cnt -= 1
            lhs_sum += yi; rhs_sum -= yi
            lhs_sum2 += yi**2; rhs_sum2 -= yi**2
            if i<self.min_leaf-1 or xi==sort_x[i+1]:
                continue

            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2) # aggregate std dev for this subtree
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt
            if curr_score<self.score: 
                self.var_idx,self.score,self.split = var_idx,curr_score,xi

    @property
    # @property so don't need () when used: self.is_leaf not self.is_leaf()
    def split_name(self): return self.x.columns[self.var_idx]
    
    @property
    def split_col(self): return self.x.values[self.idxs,self.var_idx]

    @property
    def is_leaf(self): return self.score == float('inf')
    
    # for printing
    def __repr__(self):
        s = f'n: {self.n}; val:{self.val}'
        if not self.is_leaf:
            s += f'; score:{self.score}; split:{self.split}; var:{self.split_name}'
        return s

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx]<=self.split else self.rhs
        return t.predict_row(xi)

```



# check with sklearn tree

```python
m = RandomForestRegressor(n_estimators=1, max_depth=2, bootstrap=False)
m.fit(x_samp, y_samp)
draw_tree(m.estimators_[0], x_samp, precision=2)
```

So we could see a tree: where it splits.


# run implemented model

```python
cols = ['MachineID', 'YearMade', 'MachineHoursCurrentMeter', 'ProductSize', 'Enclosure',
        'Coupler_System', 'saleYear']


ens = TreeEnsemble(X_train[cols], y_train, 5, 1000)

preds = ens.predict(X_valid[cols].values)

plt.scatter(y_valid, preds, alpha=0.1, s=6);

metrics.r2_score(y_valid, preds)

```
