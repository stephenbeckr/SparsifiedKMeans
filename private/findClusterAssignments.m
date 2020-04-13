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
% July 22, 2015 -- Aug 6 2015, May -- July 7 2016
% Updates April 2020

persistent mexFileExists_A  matlabmexFileExists newVersionMatlab
if isempty(mexFileExists_A), mexFileExists_A = 0; end
%{
We have 2 mex files that can speed things up.
Since this code is called often, we only want to check once
to see if these files exist
  mexFileExists_A   corresponds to our own mex file
    SparseMatrixMinusCluster.c
  matlabmexFileExists corresponds to a mex file written by Mathworks
    as part of their statistics toolbox, pdist2mex.c
%}
if nargin < 3, tryBuiltinMex = []; end
if isempty(tryBuiltinMex), tryBuiltinMex = true; end   % default
if nargin < 4, gamma = []; end

[p,n]   = size(X);
[pp,k]  = size(centers);
if ~isequal(p,pp), error('Array of centers not of correct size'); end

distances   = zeros( k, n );
do_sqrt     = false;
skip_min    = false;
if issparse(X)
        if mexFileExists_A || 3==exist('SparseMatrixMinusCluster','file')
            % use the fast mex code
            if issparse(centers)
                if ~isempty(gamma)
                    for ki = 1:k
                        ind     = find( centers(:,ki) );
                        gamma_center    = nnz(centers(:,ki))/size(centers,1); % changes...
                        distances( ki, : ) =SparseMatrixMinusCluster(X(ind,:)/gamma_center, full(centers(ind,ki))/gamma );
                    end
                else
                    for ki = 1:k
                        ind     = find( centers(:,ki) );
                        distances( ki, : ) =SparseMatrixMinusCluster(X(ind,:), full(centers(ind,ki)) );
                    end
                end
            else
                if ~isempty(gamma)
                    distances =SparseMatrixMinusCluster(X, centers/gamma );
                else
                    distances =SparseMatrixMinusCluster(X, centers );
                end
            end
            mexFileExists_A = true;
        else
            warning('findClusterAssigments:noMex','cannot find mex file in your path, using slower code');
            % Note: we could speed this up using matrix multiplies, but
            %   also need to find norms of columns, best done in mex.
            %   Not worrying about this code much, since you should be
            %   able to get the mex code above working.
            if issparse(centers)
                if ~isempty(gamma)
                    for ki = 1:k
                        for j = 1:n
                            ind     = find( X(:,j) );
                            ind     = intersect( ind, find(centers(:,ki) ) );
                            gamma_center    = nnz(centers(:,ki))/size(centers,1); % changes...
                            if ~isempty(ind)
                                distances(ki,j) = norm( X(ind,j)/gamma_center - centers(ind,ki)/gamma );
                            end
                        end
                    end                    
                else
                    for ki = 1:k
                        for j = 1:n
                            ind     = find( X(:,j) );
                            ind     = intersect( ind, find(centers(:,ki) ) );
                            if ~isempty(ind)
                                distances(ki,j) = norm( X(ind,j) - centers(ind,ki) );
                            end
                        end
                    end
                end
            else
                if ~isempty(gamma), centers = centers/gamma; end
                for ki = 1:k
                    for j = 1:n
                        ind     = find( X(:,j) );
                        distances(ki,j) = norm( X(ind,j) - centers(ind,ki) );
                    end
                end
            end
        end
%     end
else
    % Dense case. Optimized (May 2016)
    % Update, April 2020: let's explain what we did. It used to be that
    %   Matlab's "pdist2" code wasn't that fast, but then ~2015/2016 they
    %   updated the pdist2 code to call a mex file, so it was efficient.
    % Then, noticing that in R2019b at least (if not earlier) that it still
    % calls the efficient code, but the executable isn't a standalone mex
    % file anymore, it's buried somewhere in Matlab's general libraries, so
    % I can't call it directly. Can just call pdist2(), which is somewhat
    % fast (at least, better than my code if k is large, slower if k is
    % small)
    if tryBuiltinMex && isempty(newVersionMatlab)
        newVersionMatlab = ~verLessThan( 'matlab', '9.0'); % R2016a
        % Not sure if that's when they changed how pdist2 calls mex, since
        % not documented, but somewhat close
    end
    if tryBuiltinMex && ~any(~matlabmexFileExists)
        if isempty(matlabmexFileExists)
            matlabmexFileExists = moveMatlabMex(newVersionMatlab);
        end
    end
        
    if tryBuiltinMex && ~any(~matlabmexFileExists)
        [distances,assignments] = pdist2mex( centers, X, 'euc', [], 1, [] );
        skip_min    = true;
    elseif tryBuiltinMex && newVersionMatlab && k > 30
        % April 2020, call pdist2() itself (which calls pdist2mex)
        %   which is sometimes faster (if k is large) than my code
        [distances,assignments] = pdist2(  centers', X', 'euclidean','smallest',1 );
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
    distances = sqrt(max(0,distances));
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


function flag = moveMatlabMex(newVersionMatlab)
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
        fprintf('Added mex file to your temporary \n directory %s or it already exists (can delete later)\n', tempdir );
        flag = 1;
    end
else
    if ~newVersionMatlab
        disp('Cannot find pdist2mex -- do you have the stats toolbox?');
    end
    % (in newer versions of Matlab, this is not unexpected, so don't issue
    % a warning)
end
