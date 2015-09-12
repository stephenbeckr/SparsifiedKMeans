function [assignments,distances] = recalculateAssignmentLargeFile( fileName, centers, varargin )
% [assignments,distances] = recalculateAssignmentLargeFile( fileName, centers )
%   uses the matrix (say, "X") from fileName (a .mat file)
%   and computes assignments from X with respect to the centers
%
%   The point of this file is to never load ALL of the mat file
%       into memory, e.g., the case that the .mat file is 100 GB
%       and you only have 500 MB of RAM
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
%
% Stephen Becker
% Stephen.Becker@Colorado.edu, 9/11/2015

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


MB      = 1024^2;
% 8 bytes per double entry
% We have p rows, so allow nn columns, where nn*p*8 = MB_limit*MB
nn      = floor( MB_limit*MB/( p*8 ) );
nBlocks = ceil( n/nn );
if Verbose
    fprintf('Splitting %d x %d matrix into %d %d x %d chunks\n', ...
        p,n,nBlocks, p,nn);
end

assignments = [];
distances   = [];

for j = 1:nBlocks
    ind = (1+(j-1)*nn):min(n,j*nn);
    if ColumnSamples 
        X_ind   = matObj.(var)(:,ind);
    else
        X_ind   = (matObj.(var)(ind,:))';
    end
    
    [a_j,d_j]     = findClusterAssignments(X_ind,centers);
    assignments = [assignments,a_j];
    distances   = [distances,  d_j];
end