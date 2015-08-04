function [assignments,distances] = findClusterAssignments( X, centers )
% assignments = findClusterAssignments( X, centers )
%
%   takes a data matrix X, which is p x n,
%   and a matrix of cluster centers "centers", which is p x k,
%   and returns a 1 x n vector of cluster assignments,
%   where assignment(i) is an integer between 1 and k, corresponding
%   to which cluster X(:,i) is assigned to (based on closest
%    proximity in Euclidean norm)
%
%   If X is a sparse matrix, then considers distance based only
%    on the non-zero elements of X(:,i)
%    In this case, the code is much faster if you have
%    the mex file SparseMatrixMinusCluster.mex...
%    You can compile this like:
%       mex -largeArrayDims SparseMatrixMinusCluster.c
%
% [assignments,distances] = ...
%   also returns the distances to each cluster
%
% Stephen Becker, stephen.becker@colorado.edu
% July 22, 2015

[p,n]   = size(X);
[pp,k]  = size(centers);
if ~isequal(p,pp), error('Array of centers not of correct size'); end

distances   = zeros( k, n );

if issparse(X)
    if 3==exist('SparseMatrixMinusCluster','file') 
        % use the fast mex code
        for ki = 1:k
            distances( ki, : ) =SparseMatrixMinusCluster(X, centers(:,ki) );
        end
    else
        warning('findClusterAssigments:noMex','cannot find mex file in your path, using slower code');
        
        for ki = 1:k
            for j = 1:n
                ind     = find( X(:,j) );
                distances(ki,j) = norm( X(ind,j) - centers(ind,ki) );
            end
        end
    end
    
else
    for ki = 1:k
        differences        = bsxfun( @minus, X, centers(:,ki) );
        distances( ki, : ) = sqrt( sum(differences.^2,1) );
    end
end

[distances,assignments] = min( distances );