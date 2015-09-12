function [Y,n] = sampleAndMixFromLargeFile( fileName, SparsityLevel, P, p2, varargin )
% Y = sampleAndMixFromLargeFile( fileName, SparsityLevel, P, p2 )
%   uses the matrix (say, "X") from fileName (a .mat file)
%   and computes SparseMask.*( P(X) )
%   where
%       P is a function handle that represents matrix-vector multiplication
%       SparsityLevel controls what percentage of entries to keep
%           and in turn, this creates the "SparseMask"
%       p2  is the size of outputs from P(.)
%
%   The point of this file is to never load ALL of the mat file
%       into memory, e.g., the case that the .mat file is 100 GB
%       and you only have 500 MB of RAM
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
%
% Stephen Becker
% Stephen.Becker@Colorado.edu, 9/2/2015

prs     = inputParser;
addParameter(prs,'ColumnSamples', false );
addParameter(prs,'MB_limit',500 );
addParameter(prs,'Verbose',false );
parse( prs, varargin{:} );

ColumnSamples = prs.Results.ColumnSamples;
MB_limit      = prs.Results.MB_limit;
Verbose       = prs.Results.Verbose;

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
nn      = floor( MB_limit*MB/( p*8 ) );
nBlocks = ceil( n/nn );
if Verbose
    fprintf('Splitting %d x %d matrix into %d %d x %d chunks\n', ...
        p,n,nBlocks, p,nn);
end


Y           = [];
small_p     = round( SparsityLevel*p2 );
for j = 1:nBlocks
    ind = (1+(j-1)*nn):min(n,j*nn);
    if ColumnSamples 
        X_ind   = matObj.(var)(:,ind);
    else
        X_ind   = (matObj.(var)(ind,:))';
    end
    X_ind   = P(X_ind); % apply preconditioning
    
    nn  = length(ind);
    Yj  = spalloc(p,nn,small_p*nn);
    replace = false;
    for jj = 1:nn
        row_ind     = randsample(p2,small_p, replace );
        Yj(row_ind,jj)    = X_ind(row_ind,jj)/SparsityLevel;
    end

    Y   = [Y, Yj];
end