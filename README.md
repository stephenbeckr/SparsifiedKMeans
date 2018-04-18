# SparsifiedKMeans
KMeans for big data using preconditioning and sparsification, Matlab implementation. Uses the [KMeans clustering algorithm](https://en.wikipedia.org/wiki/K-means_clustering) (also known as [Lloyd's Algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) or "K Means" or "K-Means") but sparsifies the data in a special manner to achieve significant (and tunable) savings in computation time and memory.

The code provides `kmeans_sparsified` which is used much like the `kmeans` function from the Statistics toolbox in Matlab.
There are three benefits:

1. The basic implementation is much faster than the Statistics toolbox version. We also have a few modern options that the toolbox version lacks; e.g., we implement [K-means++](https://en.wikipedia.org/wiki/K-means%2B%2B) for initialization. (Update: Since 2015, Matlab has improved the speed of their routine and initialization, and now their version and ours are comparable).
2. We have a new variant, called sparsified KMeans, that preconditions and then samples the data, and this version can be thousands of times faster, and is designed for big data sets that are unmangeable otherwise
3. The code also allows a big-data option. Instead of passing in a matrix of data, you give it the location of a .mat file, and the code will break the data into chunks. This is useful when the data is, say, 10 TB and your computer only has 6 GB of RAM. The data is loaded in smaller chunks (e.g., less than 6 GB), which is then preconditioned and sampled and discarded from RAM, and then the next data chunk is processed. The entire algorithm is one-pass over the dataset.

/Note/: if you use our code in an academic paper, we appreciate it if you cite us:
"Preconditioned Data Sparsification for Big Data with Applications to PCA and K-means", F. Pourkamali Anaraki and S. Becker, IEEE Trans. Info. Theory, 2017.

## Why use it?
For moderate to large data, we believe this is one of the fastest ways to run k-means. For extremely large data that cannot all fit into core memory of your computer, we believe there are almost no good alternatives (in theory and practice) to this code.

# Installation
Every time you start a new Matlab session, run `setup_kmeans` and it will correctly set the paths. The first time you run it, it may also compile some mex files; for this, you need a valid `C` compiler (see http://www.mathworks.com/support/compilers/R2015a/index.html).

# Version
Current version is 2.1


# Authors
* [Prof. Stephen Becker](http://amath.colorado.edu/faculty/becker/), University of Colorado Boulder (Applied Mathematics)
* [Dr. Farhad Pourkamali Anaraki](http://www.pourkamali.com/), University of Colorado Boulder (Applied Mathematics)

# Reference
[Preconditioned Data Sparsification for Big Data with Applications to PCA and K-means](https://doi.org/10.1109/TIT.2017.2672725), F. Pourkamali Anaraki and S. Becker, IEEE Trans. Info. Theory, 2017.  See also the [arXiv version](https://arxiv.org/abs/1511.00152)

Bibtex:

    @article{SparsifiedKmeans,
	    title = {Preconditioned Data Sparsification for Big Data with Applications to {PCA} and {K}-means},
	    Author = {Pourkamali-Anaraki, F. and Becker, S.},
	    year = 2017,
	    doi = {10.1109/TIT.2017.2672725},
	    journal = {IEEE Trans. Info. Theory},
	    volume = 63,
	    number = 5,
	    pages = {2954--2974}
	}

# Related projects
* [sparsekmeans](https://github.com/EricKightley/sparsekmeans) by Eric Kightley is our joint project to implement the algorithm in python, and support out-of-memory operations.
The [sparseklearn](https://github.com/EricKightley/sparseklearn) is the generalization of this idea to other types of machine learning algorithms (also python).

# Further information
Some images taken from the paper or slides from presentations; see the journal paper for full explanations

## Example on synthetic data
![Example on synthetic data](figs/example.png?raw=true "Example on synthetic data")

## Main idea
![Main idea](figs/slides_mainIdea.jpg?raw=true "Explaining our concept")
## MNIST experiment
![MNIST experiment](figs/slides_experiment1.jpg?raw=true "Experiment 1")
![MNIST accuracy](figs/slides_experiment2.jpg?raw=true "Experiment 2")

## Infinite MNIST big data experiment
![MNIST2 accuracy](figs/slides_experiment3.jpg?raw=true "Experiment 3")

## Two-pass mode for increased accuracy
![Two pass](figs/slides_experiment4.jpg?raw=true "Experiment 4")

## Theory
![Theory](figs/slides_theory.jpg?raw=true "Theorems")
