function [assignments,distances,centers] = findClusterAssignments( X, centers )
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
% [assignments,distances,newCenters] = ...
%   also returns the new centers computed with these assignments
%   (If a cluster has nothing assigned to it, we arbitarily
%    give it a center-point of 0 )
%
% Stephen Becker, stephen.becker@colorado.edu
% July 22, 2015 -- Aug 6 2015

persistent mexFileExists
if isempty(mexFileExists), mexFileExists = 0; end

[p,n]   = size(X);
[pp,k]  = size(centers);
% centers = full(centers); % allow it to be sparse
if ~isequal(p,pp), error('Array of centers not of correct size'); end

distances   = zeros( k, n );

if issparse(X)
    if mexFileExists || 3==exist('SparseMatrixMinusCluster','file') 
        % use the fast mex code
        if issparse(centers)
            for ki = 1:k
                ind     = find( centers(:,ki) );
                distances( ki, : ) =SparseMatrixMinusCluster(X(ind,:), full(centers(ind,ki)) );
            end
        else
            for ki = 1:k
                distances( ki, : ) =SparseMatrixMinusCluster(X, centers(:,ki) );
            end
        end
        mexFileExists = true;
    else
        warning('findClusterAssigments:noMex','cannot find mex file in your path, using slower code');
        if issparse(centers)
            for ki = 1:k
                for j = 1:n
                    ind     = find( X(:,j) );
                    ind     = intersect( ind, find(centers(:,ki) ) );
                    if ~isempty(ind)
                        distances(ki,j) = norm( X(ind,j) - centers(ind,ki) );
                    end
                end
            end
        else
            for ki = 1:k
                for j = 1:n
                    ind     = find( X(:,j) );
                    distances(ki,j) = norm( X(ind,j) - centers(ind,ki) );
                end
            end
        end
    end
    
else
    for ki = 1:k
        differences        = bsxfun( @minus, X, centers(:,ki) );
        distances( ki, : ) = sqrt( sum(differences.^2,1) );
    end
end

[distances,assignments] = min( distances, [], 1 );

if nargout >= 3
    % user has requested new centers
    for ki = 1:k
        ind             = find( assignments == ki );
        if isempty(ind)
            centers(:,ki)   = 0;
        else
            centers(:,ki)   = mean( full(X(:, ind ) ), 2 );
        end
    end
end