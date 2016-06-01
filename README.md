# SparsifiedKMeans
KMeans for big data using preconditioning and sparsification, Matlab implementation

The code provides `kmeans_sparsified` which is used much like the `kmeans` function from the Statistics toolbox in Matlab.
There are three benefits:

1. The basic implementation is much faster than the Statistics toolbox version. We also have a few modern options that the toolbox version lacks; e.g., we implement [K-means++](https://en.wikipedia.org/wiki/K-means%2B%2B) for initialization.
2. We have a new variant, called sparsified KMeans, that preconditions and then samples the data, and this version can be thousands of times faster, and is designed for big data sets that are unmangeable otherwise
3. The code also allows a big-data option. Instead of passing in a matrix of data, you give it the location of a .mat file, and the code will break the data into chunks. This is useful when the data is, say, 10 TB and your computer only has 6 GB of RAM. The data is loaded in smaller chunks (e.g., less than 6 GB), which is then preconditioned and sampled and discareded from RAM, andn then the next data chunk is processed. The entire algorithm is one-pass over the dataset.

# Installation
Every time you start a new Matlab session, run `setup_kmeans` and it will correctly set the paths. The first time you run it, it may also compile some mex files; for this, you need a valid `C` compiler (see http://www.mathworks.com/support/compilers/R2015a/index.html).

# Version
Current version is 2.1

# Example
![Example on synthetic data](example.png?raw=true "Example on synthetic data")

# Authors
* [Stephen Becker](http://amath.colorado.edu/faculty/becker/), University of Colorado Boulder (Applied Mathematics)
* [Farhad Pourkamali Anaraki](http://www.pourkamali.com/), University of Colorado Boulder (Electrical Engineering)
