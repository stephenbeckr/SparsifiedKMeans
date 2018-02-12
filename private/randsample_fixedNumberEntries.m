function Y = randsample_fixedNumberEntries( X, small_p, varargin )
% Y = randsample_fixedNumberEntries( X, small_p )
%   returns a sparse matrix Y which has exactly "small_p" number of
%   nonzeros per column; the nonzero entries of each column are chosen
%   uniformly at random (and each column is independent) from the entries
%   of the corresponding column of X (and scaled by p/small_p)
%
%   This code is designed to be efficient (for 600,000 x 748 matrix,
%       takes about 6 seconds instead of 130 seconds with naive
%       implementation).
%
%  Y = randsample_fixedNumberEntries( X, small_p, 'MB_limit', 300 )
%       will set the memory limit to, e.g., 300 MB. This is used
%       to determine the block-size of the algorithm. Default is 100 MB.
%
% See also (this function relies on this code):
%   RANDSAMPLE_BLOCK.m

% Stephen.Becker@Colorado.edu, 6/13/2016

prs     = inputParser;
addParameter(prs,'ColumnSamples', true );
addParameter(prs,'MB_limit',100 );
parse( prs, varargin{:} );

ColumnSamples = prs.Results.ColumnSamples;
MB_limit      = prs.Results.MB_limit;
if ~ColumnSamples, error('not configured for row samples'); end

[p,n]   = size(X);
SparsityLevel   = small_p/p;

% ** Old, slow code, but shows you what we are doing **
% Y           = spalloc(p,n,small_p*n);
% replace = false;
% for j = 1:n
%     ind         = randsample(p,small_p, replace );
%     Y(ind,j)    = X(ind,j)/SparsityLevel;
% end

% ** Newer, faster code **

% bottleneck memory is indBig which is small_p x blk, 8 bytes per double entry
MB          = 1024^2;
blockSize   = floor( MB_limit*MB/( small_p*8 ) );
blockSize   = min( blockSize, n );

Y = [];
for j = 1:ceil( n/blockSize )
    if j < ceil( n/blockSize )
        blk     = blockSize;
    else
        blk     = n - (j-1)*blockSize;
    end
    offset  = (j-1)*p*blockSize;
    indBig  = randsample_block( p, small_p, blk );
    J       = repmat( 1:blk, small_p, 1 );
    % and convert to linear, with offset
    indBigLinear = bsxfun( @plus, indBig, offset + p*(0:blk-1) );
    %Yj      = sparse(indBig,J(:), X(indBigLinear(:))/SparsityLevel ); %
    %Buggy, 2/11/18. Fix is below:
    Yj      = sparse(indBig,J(:), X(indBigLinear(:))/SparsityLevel, size(X,1), blk );
    Y       = [Y, Yj];
end