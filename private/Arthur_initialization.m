function centers = Arthur_initialization( X, K )
% centers = Arthur_initialization( X, K )
%   finds K centers to initialize K-means with
%   This is the so-called K-means++ initialization
%
%   Note: expects X to be p x n, with datapoints as columns, not rows
%
% K-means++ initialization: see
%   https://en.wikipedia.org/wiki/K-means%2B%2B
%   Arthur, D. and Vassilvitskii, S. (2007). 
%   "k-means++: the advantages of careful seeding"  SODA pp. 1027?1035.
% similar to:
% R. Ostrovsky, Y. Rabani, L. J. Schulman, and C. Swamy. 
%   "The effectiveness of Lloyd-type methods for the k-means problem"
%   In FOCS, pages 165?176, 2006.
%
% For faster implementation when k large, try:
% http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf
%
% Stephen.Becker@Colorado.edu  Aug 5 2015
if K<1, error('K must be >= 1'); end
[p,n]   = size(X);

% First center chosen uniformly at random:
i           = randi( n, 1 );
chosenInd   = i;
centers     = X(:,i);
for k = 1:(K-1)
    [~, dist] = findClusterAssignments( X, centers );
    % now choose a new point with non-uniform probability,
    %   based on dist^2
    % Since we only need one point, we can sample with or without
    %   replacement
    if norm(dist)>0
        i   = randsample( n, 1, true, dist.^2 );
    else
        i   = randsample( n, 1, true);
    end
    counter     = 1;
    while ismember( i, chosenInd ) && counter < 20
        if norm(dist)>0
            i   = randsample( n, 1, true, dist.^2 );
        else
            i   = randsample( n, 1, true);
        end
        counter     = counter + 1;
    end
    if counter==2
        error('This should never happen; please debug');
    end
    
    chosenInd = [chosenInd, i ];
    centers   = [centers, X(:,i)];
end
    

%{
Choose one center uniformly at random from among the data points.
For each data point x, compute D(x), 
  the distance between x and the nearest center that has already been chosen.
Choose one new data point at random as a new center, 
  using a weighted probability distribution where a point x is 
  chosen with probability proportional to D(x)^2.
Repeat Steps 2 and 3 until k centers have been chosen.
%}
