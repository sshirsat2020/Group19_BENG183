# Introduction to K-Means Clustering
### By Shruti Shirsathe, Minh-Son Tran, and Lily Steiner
## Overview

k-Means Clustering is a topic that falls under Machine Learning. To give some context, there are different types of machine learning algorithms such as:
Supervised Machine Learning, Unsupervised Machine Learning, Semi-Supervised Machine Learning, and Reinforcement Learning

The two main types, Supervised and Unsupervised Machine Learning are defined as the following:
* **Supervised:** Algorithm that requires an input of data, and labels for classification. In the end, it is used to make predictions and classify data.
* **Unsupervised:** Algorithm that requires only an input of data and no labels. It is used to understand the relationships between points in the dataset.

<img src="img/unsupervised-learning.png" width="600" height="300" />

[1] In this image, the input data includes information with different animals without any labels for what they are; the unsupervised algorithm takes this input and groups the data into 3 different clusters based on how closely the data is related to one another

K-Means clustering is an unsupervised algorithm, meaning that the goal is to look for patterns in a dataset without pre-existing labels. 
Applications are to either:
1. confirm any assumptions about the types of groups that exist in the data
2. identify unknown groups in the data

<img src="img/kmeans.png" width="600" height="300" />

* * *

## The Algorithm
![Algorithm](img/algorithm.png)

1. Choose a value K as the number of cluster centers and set the cluster centers randomly. One way to choose K is by using the elbow method
2. Perform K-means clustering with different values of K. For each k, we calculate average distances to the centroid across all data points.
3. Plot these points and find the point where the average distance from the centroid falls suddenly (“Elbow”)
4. Now that we chose k and the initial centroids are chosen, we then 
calculate the distances between all the points in the data and the centroids, then group the points with the cluster center they are closest to.
5. Now we recalculate the centroid of these new clusters by finding the new center of gravity of the clusters; then group the data points to the new nearest centroid as we did before. 
6. We then repeat these steps until the centroid positions remain the same; if so, the algorithm has completed and you’ve found your clusters.

## Stopping Criteria
As mentioned above, the algorithm typically terminates when centroid positions remain the same from one iteration to the next. In terms of the data, 
this is equivalent to the all datapoints remaining within the same cluster after an iteration, as their center of gravity would remain the same.

However, the algorithm is not guaranteed to reach this termination point (or convergence point). K-means clustering is designed to approximate local
minima for the optimization of squared distances from centroids. It is not an exhaustive algorithm - the total number of clustering solutions is equivalent
to the number of ways to partition n datapoints into k clusters. This is on the order of n to the k, in other words, extremely large. In certain edge cases, if
centroid selection is nondeterministic for equal distances, the algorithm can oscillate between two solutions infinitely. As well as this, developers might
see it fit to not run the algorithm for an excessively long time. Therefore, we can introduce a maximum iterations parameter as an additional stopping criterion.

Summarily, we have 3 stopping criteria:

1. Centroids do not move
2. Clusters do not change
3. Maximum iterations have been attempted

## Algorithm in Practice

We can visualize how the algorithm functions on the following dataset. We have 2 variables, X and Y, across 25 observations shown below. The dataset has been
visualized as a 25 point scatterplot.

<img src="img/dataset.png" width="1200" height="500" />

Let's look at the first iteration of the algorithm

<img src="img/Iteration1.png" width="1200" height="500" />

1. Three centroids have been selected randomly from our dataset
2. Datapoints are partitioned into clusters based on which centroid they are nearest to. For a given datapoint $D$, the distance to a centroid $C$ is calculated by:
$$dist = \sqrt{(D.x - C.x)^2 + (D.y - C.y)^2}$$
3. Centroids move from their initial position to a new one, determined by the mean location of the datapoints within their cluster.
For a given cluster of $n$ datapoints, the coordinate position of the new centroid is given by:
$$\left( {{\displaystyle\sum_{1}^{n}D.x_n} \over n}, {{\displaystyle\sum_{1}^{n}D.y_n} \over n} \right)$$
where $D.x_n$ and $D.y_n$ are the $x$ and $y$ coordinates of the $n^{th}$ datapoint respectively.
4. We have our new centroids and are ready to iterate again

Here are the second, third, and fourth iterations, though not to the same depth shown above.

<img src="img/iteration2.png" width="450" height="225" />
<img src="img/iteration3.png" width="450" height="225" />
<img src="img/iteration4.png" width="450" height="225" />

Note that between the third and fourth iterations, none of our datapoints have moved into a new cluster. Therefore, we've satisfied our stopping criteria and the algorithm terminates. We've now successfully partitioned the 25 datapoints into one of either the red, green, or blue groups.

* * *

## Packages and Tools to Implement K Means
Python → scikit-learn package

```python
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
plt.scatter(s,x, c=kmeans.labels_)
plt.show()
```
<img src="img/iteration2.png" width="450" height="225" />
Optional params include: algorithm type, max iterations, seed count (and more)

R → built-in function
Output is a summary of the means, can further visualize clusters with fviz_cluster package
