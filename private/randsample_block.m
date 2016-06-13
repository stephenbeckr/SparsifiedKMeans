function y = randsample_block(n, k, nRep)
%RANDSAMPLE Random sample, with or without replacement.
%   Y = RANDSAMPLE_BLOCK(N,K) returns Y as a column vector of K values sampled
%   uniformly at random, without replacement, from the integers 1:N.
%   In this mode, this is the same as RANDSAMPLE
%       (Note that REPLACE is false)
%
%   Y = RANDSAMPLE_BLOCK(N,K,R) returns Y as a K x R matrix, each column
%   independently sampled (without replacement). It is designed to be
%   more efficient than just looping through the columns.
%       If 4*k > n, then this code is NOT more efficient than just
%       looping through columns.
%
%   Examples:
%
%   Draw a single value from the integers 1:10.
%      n = 10;
%      x = randsample(n,1);
%
%  NOTE: this does NOT randomly permute the output, the way RANDSAMPLE
%  does. This is often unnecessary, and quite slow.
%
%   See also RANDSAMPLE

% Stephen.Becker@Colorado.edu, 6/13/2016



nargs = nargin;

if nargs < 2
    error(message('randsample_block:TooFewInputs'));
end


% Sample without replacement
if k > n
    error(message('randsample_block:SampleTooLarge', n));
end

% If the sample is a sizable fraction of the population,
% just randomize the whole population (which involves a full
% sort of n random values), and take the first k.
if 4*k > n
    y   = zeros(k,nRep);
    for r = 1:nRep
        rp = randperm(n);
        y(:,r) = rp(1:k);
    end
    
    % If the sample is a small fraction of the population, a full sort
    % is wasteful.  Repeatedly sample with replacement until there are
    % k unique values.
else
%     x = zeros(1,n); % flags
%     sumx = 0;
%     while sumx < floor(k) % prevent infinite loop when 0<k<1
%         x(randi(n,1,k-sumx)) = 1; % sample w/replacement
%         sumx = sum(x); % count how many unique elements so far
%     end
    %sumx = zeros(1,nRep);
    x    = zeros(n,nRep,'int8');
    % Here is the big savings: we do one before the loop
    %x( randi(n,nRep,k) ) = 1;
    % above line is not valid. We need to convert to linear indices
    ind = randi(n,k,nRep); % IMAX, M, N
    ind = bsxfun( @plus, ind, n*(0:nRep-1) );
    x( ind(:) ) = 1;
    
    sumxBig    = sum(x);
    % and a few more...
    for r = 1:nRep
        sumx = sumxBig(r);
        while any( sumx < floor(k) )
            x(randi(n,1,k-sumx),r) = 1; % sample w/replacement
            sumx = sum(x(:,r)); % count how many unique elements so far
        end
    end
    
    %y = find(x > 0); % by design, each column has k nnz
    
    % Modify
    [y,~] = find(x > 0);
    y     = reshape( y, k, nRep );
    
    % This step is usually unnecessary!
    %y = y(randperm(k));
%     for r = 1:nRep
%         rp = randperm(k);
%         y(:,r) = y(rp);
%     end
end
