# Possibilistic Fuzzy Neural Gas
### Neural Gas
 Neural Gas by [T. Martinitz](http://ftp.ks.uiuc.edu/Publications/Papers/PDF/MART91B/MART91B.pdf) is inspired by the Self-Organizing Map for finding optimal data representations based on feature vectors. The algorithm was named **Neural Gas** because of the dynamics of the feature vectors during the adaptation process, which distribute themselves like a gas within the data space. Among the variety of methods that has been developed for clustering, Neural gas uses *prototype-based* approach. Prototype-based technique tries to define clusters using small set of prototypes. Where each prototype tries to represent the group of data points based on similarity measure to the prototype which can be influenced by size and shape of the parameter. Example of prototype based clustering algorithms are:
 
 - c-Means or k-means (where, c or k is the number of prototypes).
 - Self-Organizing Map
 - Neural Gas
 
 The principle version of all these clustering methods is that each data point is represented uniquely by exactly one prototype. This is also known as crisp clustering. In real world most of the time data is overlapping, so a clustering method which uses soft assginment of data point to the prototypes can be very helpful in understanding the structure of the data. This is also known as *fuzzy* clustering. Example of fuzzy clustering algorithms are: 

 - Fuzzy c-means
 - Fuzzy Self Organizing Map
 - Fuzzy Neural Gas
 
 Possibilistic Fuzzy Neural Gas(PFNG) is an extended method of the [Possibilistic Fuzzy C-Means](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1492404). There are two kind of a assignments that are being used:
  - *Probabilistic* (or Membership Value): which ranges [0, 1] and for each data point to all the prototypes the values should sums up to 1.
  - *Possibilistic* (or Typicality Values): the value of the assignment decreases with the increase in the distance to the prototype.

PFNG can be used to cluster different kind of data:
 - Vectorial data: Contains n-dimensional real valued vectors, and the distance measures used here is squared Euclidean distance. It is defined as 
 -                                              D<sub>ij;</sub> = d^2<sub>ij;</sub> 
