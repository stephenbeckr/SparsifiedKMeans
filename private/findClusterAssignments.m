function [assignments,distances,centers] = findClusterAssignments( X, centers, tryBuiltinMex, gamma )
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
% ... = findClusterAssignments( X, centers, tryBuiltinMex )
%   will try using Matlab's pdist2mex mex file which exists in some
%   newer versions of Matlab, if tryBuiltinMex is true (default)
%   This only affects the case when both X and centers are dense.
%
% ... = findClusterAssignments( X, centers, tryBuiltinMex, gamma )
%   will use newer code to estimate distances; only affects
%   the case when X is sparse. Set gamma=[] to turn off (default)
%
% Stephen Becker, stephen.becker@colorado.edu
% July 22, 2015 -- Aug 6 2015

persistent mexFileExists_A mexFileExists_B  matlabmexFileExists
if isempty(mexFileExists_A), mexFileExists_A = 0; end
if isempty(mexFileExists_B), mexFileExists_B = 0; end
%{
We have 3 mex files that can speed things up.
Since this code is called often, we only want to check once
to see if these files exist
  mexFileExists_A   corresponds to our own mex file
    SparseMatrixMinusCluster.c
  mexFileExists_B   corresponds to our own mex file
    SparseMatrixInnerProduct.c
  matlabmexFileExists corresponds to a mex file written by Mathworks
    as part of their statistics toolbox, pdist2mex.c
%}
if nargin < 3, tryBuiltinMex = []; end
if isempty(tryBuiltinMex), tryBuiltinMex = true; end   % default
if nargin < 4, gamma = []; end

[p,n]   = size(X);
[pp,k]  = size(centers);
% centers = full(centers); % allow it to be sparse
if ~isequal(p,pp), error('Array of centers not of correct size'); end

LARGE_K     = 2; % controls behavior of code

distances   = zeros( k, n );
do_sqrt     = false;
skip_min    = false;
if issparse(X)
    if ~isempty(gamma)
        if mexFileExists_B || 3==exist('SparseMatrixInnerProduct','file')
            % use the fast mex code
            if issparse(centers)
                normX2    = SparseMatrixColumnNormSq(X); % do just once
                normC2    = SparseMatrixColumnNormSq(centers);
                crossTermAll   = ( X'*centers )';
                for ki = 1:k
                    gamma_center    = nnz(centers(:,ki))/size(centers,1); % changes...
                    distances( ki, : ) = bsxfun(@plus, gamma*normX2 - 2*crossTermAll(ki,:) , gamma_center*normC2(ki) );
                end
                
                % Old code (and slightly different calculation,
                %  likely to lead to zeros if both vectors very sparse)
%                 for ki = 1:k
%                     ind     = find( centers(:,ki) );
%                     gamma_center    = length(ind)/size(centers,1); % changes...
%                     [crossTerm,normX2] = SparseMatrixInnerProduct( X(ind,:), full(centers(ind,ki)) );
%                     distances( ki, : ) = bsxfun(@plus, gamma*normX2 - 2*crossTerm , gamma_center*norm(full(centers(ind,ki)))^2 );
%                 end
            else
                if k <= LARGE_K
                    for ki = 1:k
                        [crossTerm,normX2] = SparseMatrixInnerProduct( X, centers(:,ki) );
                        distances( ki, : ) = bsxfun(@plus, gamma*normX2 - 2*crossTerm , norm(centers(:,ki))^2 );
                    end
                else
                    % do batched operations
                    normX2    = SparseMatrixColumnNormSq(X); % do just once
                    crossTermAll   = ( X'*centers )';
                    for ki = 1:k
                        distances( ki, : ) = bsxfun(@plus, gamma*normX2 - 2*crossTermAll(ki,:) , norm(centers(:,ki))^2 );
                    end
                end
            end
            mexFileExists_B = true;
            do_sqrt     = true;
            distances   = max( distances, 0 ); % makes it biased though...
        else
            error('not yet implemented');
        end
    else
        
        if mexFileExists_A || 3==exist('SparseMatrixMinusCluster','file')
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
            mexFileExists_A = true;
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
    end
else
    % Dense case. Optimized (May 2016)
    if tryBuiltinMex && ~any(~matlabmexFileExists)
        if isempty(matlabmexFileExists)
            matlabmexFileExists = moveMatlabMex();
        end
    end
    if tryBuiltinMex && ~any(~matlabmexFileExists)
        [distances,assignments] = pdist2mex( centers, X, 'euc', [], 1, [] );
        skip_min    = true;
    else
        % May 2016, expand quadratic
        nrm2    = sum( X.^2, 1 ); % pre-calculate
        for ki = 1:k
            distances( ki, : ) = nrm2 -2*(X'*centers(:,ki))' + norm(centers(:,ki))^2;
            %         distances( ki, : ) = nrm2 -2*(centers(:,ki)'*X) + norm(centers(:,ki))^2; % same speed
            % Older, slower code, prior to May 2016
            %         differences        = bsxfun( @minus, X, centers(:,ki) );
            %         distances( ki, : ) = sqrt( sum(differences.^2,1) );
        end
        do_sqrt = true;
    end
end

if ~skip_min
    [distances,assignments] = min( distances, [], 1 );
    % (for pdist2mex, we already took the min)
end
if do_sqrt
    % only do this on the minimum, not on all
    distances = sqrt(distances);
end


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


function flag = moveMatlabMex()
% successFlag = moveMatlabMex()
%   successFlag = 1 means that it worked

mxFile  = fullfile(matlabroot,'toolbox','stats','stats','private',['pdist2mex.' mexext()] );
flag    = 0; % fail, by default
if 3==exist(mxFile,'file')
    % the user may not have the stats toolbox
    % we have to copy somewhere else since Matlab won't read it
    %   if it is in a "private" subdirectory
    [success,message,messageID] = copyfile(mxFile, tempdir );
    if success==1 || strcmpi(messageID,'MATLAB:COPYFILE:ReadOnly')
        addpath(tempdir)
        fprintf('Added mex file to your temporary \n directory %s or it already exists(can delete later)\n', tempdir );
        flag = 1;
    end
else
    disp('Cannot find pdist2mex -- do you have the stats toolbox?');
end