function [Y,n,timeMix,timeSample] = sampleAndMixFromLargeFile( fileName, SparsityLevel, P, p, varargin )
% Y = sampleAndMixFromLargeFile( fileName, SparsityLevel, P, p )
%   uses the matrix (say, "X") from fileName (a .mat file)
%   and computes SparseMask.*( P(X) )
%   where
%       P is a function handle that represents matrix-vector multiplication
%       SparsityLevel controls what percentage of entries to keep
%           and in turn, this creates the "SparseMask"
%       p  is the size of outputs from P(.)
%
%   The point of this file is to never load ALL of the mat file
%       into memory, e.g., the case that the .mat file is 100 GB
%       and you only have 500 MB of RAM
%
% [Y,timeLoad,timeMix,timeSample] = sampleAndMixFromLargeFile( fileName, SparsityLevel, ... )
%       returns the time to load the data and the time to mix the data.
%
% [p,n] = sampleAndMixFromLargeFile( fileName, 0, [], [] )
% [p,n] = sampleAndMixFromLargeFile( fileName, 0, [], [], 'ColumnSamples',true)
%   will not load any data, just return the dimension p
%
% IMPORTANT: the .mat file MUST be saved with -v7.3 format,
%   which is NOT default.
%
% The following are parameter/value [default] pairs that you can pass in:
%   'MB_limit'  [500]     Controls how much of the file to load
%                           at once. Set this to whatever RAM you
%                           can afford
%   'Verbose'   [false]   Displays more output if true
%   'ColumnSamples' [false]   If false, then each sample/entry is a row,
%                               and if true, then each sample/entry
%                               is a column (e.g., X is p x n)
%                             NOTE: regardless of this value,
%                               the output Y is p x n
%   'AddEps'    [true]    Adds a tiny change to the data to make zeros
%                           into nonzeros so as not to trigger sparse
%                           code (which interprets zeros as missing data,
%                           not just as zero value).
%                         Setting this to false can ruin the sampling,
%                           but if your data are sparse, setting this to
%                           true will make code run slowly. If data are
%                           sparse, you should not be sampling though!
%
% Stephen Becker
% Stephen.Becker@Colorado.edu, 9/2/2015; updated 6/13/2016

prs     = inputParser;
addParameter(prs,'ColumnSamples', false );
addParameter(prs,'MB_limit',500 );
addParameter(prs,'Verbose',false );
addParameter(prs,'AddEps',true); % 6/1/2016
parse( prs, varargin{:} );

ColumnSamples = prs.Results.ColumnSamples;
MB_limit      = prs.Results.MB_limit;
Verbose       = prs.Results.Verbose;
AddEps        = prs.Results.AddEps;

% -- Find the file:
matObj  = matfile( fileName, 'Writable', false );
varNames = fieldnames(matObj);
if length(varNames) > 2, error('Expected a single variable'); end
if strcmpi(varNames{1},'Properties'), var = varNames{2}; else var = varNames{1}; end

[p,n]   = size( matObj, var );
if ~ColumnSamples % switch dimensions...
    pTemp = p;
    p     = n;
    n     = pTemp;
end

% Special behavior: with only
if SparsityLevel==0
    Y = p;
    return;
end


MB      = 1024^2;
% 8 bytes per double entry
% We have p rows, so allow nn columns, where nn*p*8 = MB_limit*MB
nn      = max(1,floor( MB_limit*MB/( p*8 ) )); % 2/11/18, fixed "min" bug
nn      = min( nn, n );
nBlocks = ceil( n/nn );
if Verbose
    fprintf('Splitting %d x %d matrix into %d %d x %d chunks\n', ...
        p,n,nBlocks, p,nn);
    if nn == 1 && nBlocks > 500
        fprintf('Warning: there are a lot of blocks, this may be slow. Is MB_limit set correctly?');
    end
end


Y           = [];
small_p     = round( SparsityLevel*p );
 SparsityLevel  = small_p/p; % undo rounding. 6/13/2016
timeLoad    = 0;
timeMix     = 0;
timeSample  = 0;
for j = 1:nBlocks
    ind = (1+(j-1)*nn):min(n,j*nn);
    t1 = tic;
    if ColumnSamples 
        X_ind   = matObj.(var)(:,ind);
    else
        X_ind   = (matObj.(var)(ind,:))';
    end
    timeLoad    = timeLoad + toc(t1);
    if AddEps
        X_ind   = X_ind*(1 + 2*eps); % eps is machine epsilon (6/1/2016)
    end
    t1      = tic;
    X_ind   = P(X_ind); % apply preconditioning
    timeMix = timeMix + toc(t1);
    
    t1      = tic;
    %nn  = length(ind);
    %Yj  = spalloc(p,nn,small_p*nn);
    %replace = false;
    %for jj = 1:nn
        %row_ind     = randsample(p,small_p, replace );
        %Yj(row_ind,jj)    = X_ind(row_ind,jj)/SparsityLevel;
    %end
    % 6/13/2016, more efficient:
    Yj = randsample_fixedNumberEntries( X_ind, small_p );

    Y   = [Y, Yj];
    timeSample  = timeSample + toc(t1);
end
n = timeLoad;
