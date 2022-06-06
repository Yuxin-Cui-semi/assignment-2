# assignment-2

This assignment can be described as three main steps

First step is to get the timeseris dataset for six states:
1, generate the grid from given file by using pandapower
2, define six different work state
3, get the timeseries from pandapower (inspired by EH2745 GitHub repository L14)
4, integrate all the dataset for Kmeans and KNN

Second step is to clusting the data by using Kmeans algorithm
1, initial the centroid and do clusting
2, update the centroids
3, do clusting again, repeat until the distance is lower than epsilon or the iteration time equals to the maxstep
4, visualize the result of clusting

Final step is to test data by using KNN algorithm
1, calculate distance between test point and train points
2, sort the distance from small to large
3, choose number of K neareast points
4, check how many time are the neareast points showing up
5, get the result and check the accuracy

Thank you.
