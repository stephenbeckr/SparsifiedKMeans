# SparsifiedKMeans
KMeans for big data using preconditioning and sparsification, Matlab implementation

The code provides `kmeans_sparsified` which is used much like the `kmeans` function from the Statistics toolbox in Matlab.
There are two benefits:

1. The basic implementation is much faster than the Statistics toolbox version
2. We have a new variant, called sparsified KMeans, that preconditions and then samples the data, and this version can be thousands of times faster, and is designed for big data sets that are unmangeable otherwise

# Installation
Every time you start a new Matlab session, run `setup_kmeans` and it will correctly set the paths. The first time you run it, it may also compile some mex files; for this, you need a valid `C` compiler (see http://www.mathworks.com/support/compilers/R2015a/index.html).


# Example
![Example on synthetic data](example.png?raw=true "Example on synthetic data")

# Authors
* [Stephen Becker](http://amath.colorado.edu/faculty/becker/), University of Colorado Boulder (Applied Mathematics)
* [Farhad Pourkamali Anaraki](http://www.pourkamali.com/), University of Colorado Boulder (Electrical Engineering)
