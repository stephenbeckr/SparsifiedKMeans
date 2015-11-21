function [bestAssignments, bestCenters, SUMD, bestDistances, centers_twoPass] = kmeans_sparsified(X, K, varargin)
% [IDX, C, SUMD, D, C_twoPass] = kmeans_sparsified(X, K)
%
% Note: to maintain compatability with Matlab, we assume the points
%   are ROWS of X, so X is a n x p matrix
% If you want columns to be the points, then set 'ColumnSamples',true
%
% IMPORTANT: to run the "fast" (sparsified) code, make sure to 
%   set 'Sparsify',true
%
% Outputs:
%
% IDX is the index, between 1 and K, that indicates which cluster
%   each sample belongs to.
% C is the p x K matrix of cluster centers (aka centroids)
% SUMD is a 1 x K vector where each entry is the sum-of-squared-distances
%   of that cluster to its cluster members. sum(SUMD) is the global
%   objective. Note: this is calculated using D, not D_twoPass.
% D returns the distance from each point to every centroid
% C_twoPass -- if this output is requested, then this returns
%   an improved version of C that uses a second pass through the
%   entire dataset. For extremely large datasets, this will be slow.
%   If 'Sparsify' is set to false, then C_twoPass is the same as C.
%
% Examples:
%   [IDX,C] = kmeans_sparsified( X, 5 )
%   [IDX,C] = kmeans_sparsified( X', 5,'ColumnSamples',true,'tol',1e-4)
%
% Most of the options are the same as Matlab
%   D returns the Euclidean distance (e.g., sqrt( sum_i x_i^2 ) )
%   while SUMD returns a vector of squared Euclidean distances
% Options:
%   'Replicates', r         Run K-Means r times, keeping the best
%   'Start'                 if 'sample', picks K random data points
%                           if 'uniform', picks K points uniformly from
%                               the space
%                           if 'Arthur' or '++', uses Arthur/Vassilvitskii 2007
%                               K-means++ algorithm to initialize
%                           if a matrix, then use this as starting point
%                           (K x p, or 'ColumnSamples' is true, then p x K)
%   'MaxIter'               How many iterations to run for each trial
%                               (default: 100)
%   'Display'               either 'off' (default),'iter' or 'final'
%   'PrintEvery', u         if Display='iter', then prints every u
%                               iterations
%   'Tol'                   stopping tolerance (default: 1e-6)
%   'EmptyAction'           What to do when a cluster loses members
%                               'Singleton' (default),'Error' or 'Drop
%                               We also throw a warning, which can
%                               be turned off with:
%                           warning('off','kmeans_sparsified:dropCluster');
%
% Interesting options (here we depart from Matlab)
%   'Sparsify'              Run the fast sparsified version (default:
%                               false)
%   'SparsityLevel'         How sparse (between 0 and 1, close to 0
%                               being the most extreme)
%   'SketchType'            Either 'none' or 'DCT' (default) or 'Hadamard'
%                               Controls the pre-conditioning
%                               (Defaults to Hadamard if p is a power of 
%                                2).
%   'MLcorrection'          Theoretical normalization; keep this true
%                               (default). Included only for testing
%   'ColumnSamples'         If false (default), input is n x p, i.e.,
%                               samples as rows. If true, input is p x n
%                               and the output is also transposed.
%
%   'DataFile'              If this is true and a valid file name of a .mat file,
%                               then matlab loads the file and will read
%                               in parts of the file at a time. This is
%                               useful when the file is larger in size
%                               than the RAM of your computer. The file
%                               is broken into as few pieces as possible
%                               such that each piece is smaller than
%                               MB_limit
%       IMPORTANT: the .mat DataFile MUST be saved with -v7.3 format,
%       which is NOT default.
%   'MB_limit'              The limit, in megabytes (MB), of how large
%                               each piece of the main file can be.
%                               Used only with the 'DataFile' option
%   'DataFileVerbose'       If true (default is false), tells you
%                               how many chunks the file is broken into.
%   'SparsityIgnoreUpsampling' If true (default is false), then if using
%                              a sketch like Hadamard which may upsample,
%                              we ignore the new larger upsampled dimension
%                              when calculating sparsity. So this will
%                              result in faster computation but worse
%                              accuracy.
%   

% Stephen Becker and Farhad Pourkamali-Anaraki
% Stephen.Becker@Colorado.edu, 8/5/2015 -- 9/11/2015
% see https://github.com/stephenbeckr/SparsifiedKMeans

p = inputParser;
addParameter(p,'Replicates',1);
% expectedStart = {'sample','uniform','Arthur','k-means++','++'};
% addParameter(p,'Start','Arthur',@(x) any(validatestring(x,expectedStart)));
addParameter(p,'Start','Arthur' ); % allow matrix too for warm-start
addParameter(p,'MaxIter',100);
validDispActions={'off','iter','final'};
addParameter(p,'Display',false,@(x) any(validatestring(x,validDispActions)));
addParameter(p,'PrintEvery',10);
addParameter(p,'Tol',1e-6);
addParameter(p,'Sparsify',false);
addParameter(p,'SparsityLevel',0.01, @(x) x>0 && x<= 1 );
addParameter(p,'SketchType','auto'); % Hadamard or DCT
validEmptyActions={'singleton','error','drop'};
addParameter(p,'EmptyAction','singleton',@(x) any(validatestring(x,validEmptyActions) ));
addParameter(p,'ColumnSamples',false); 
addParameter(p,'MLcorrection',true); % normalization according to maximum likelihood derivation
addParameter(p,'DataFile',[]);
addParameter(p,'MB_limit',500 ); % only used if reading from disk
addParameter(p,'DataFileVerbose',false);
addParameter(p,'SparsityIgnoreUpsampling',false); % added 10/7/15
addParameter(p,'FORCE_BUG',false); % added 11/21/15
parse(p,varargin{:});

Replicates  = p.Results.Replicates;
start       = p.Results.Start;
MaxIter     = p.Results.MaxIter;
Display     = p.Results.Display;
tol         = p.Results.Tol;
Sparsify    = p.Results.Sparsify;
SparsityLevel    = p.Results.SparsityLevel; % used when Sketching
SketchType  = p.Results.SketchType; % Hadamard or DCT or Nothing
EmptyAction = p.Results.EmptyAction;
PrintEvery  = p.Results.PrintEvery;
ColumnSamples = p.Results.ColumnSamples;
DataFile    = p.Results.DataFile;
MB_limit    = p.Results.MB_limit;
MLcorrection= p.Results.MLcorrection && Sparsify;
FORCE_BUG   = p.Results.FORCE_BUG; % do NOT turn on unless you are testing
SparsityIgnoreUpsampling = p.Results.SparsityIgnoreUpsampling;
DataFileVerbose = p.Results.DataFileVerbose;

% Do we load X from disk?
LoadFromDisk    = ~isempty(DataFile) && 2==exist(DataFile,'file');

if LoadFromDisk
    if ~Sparsify
        error('No reason to turn on "LoadFromDisk" option if not sampling');
    end
    if ~isempty(X)
        warning('Loading data from disk, ignoring "X" input. Are you sure code is OK?');
    end
    [p,n]   = sampleAndMixFromLargeFile( DataFile, 0, [], [], 'ColumnSamples',ColumnSamples);
else
    if ~ColumnSamples
        X   = X';
    end
    [p,n]   = size(X);
end

if n<K
    error('kmeans_sparsified:badDimensions','X must have more samples than the number of clusters.');
end

p2          = p;
if Sparsify
    if strcmpi(SketchType,'auto')
        if p == 2^nextpow2(p)
            SketchType = 'Hadamard';
        else
            SketchType = 'DCT';
        end
    end
    
    if strcmpi(Display,'iter') || strcmpi(Display,'final')
       fprintf('Randomly mixing of type %s\n', SketchType );
    end
    upsample    = @(x) x;
    downsample  = upsample;
    if strcmpi( SketchType, 'Hadamard' )
        p2  = 2^nextpow2(p);
        if p<p2
            upsample = @(x) [x;zeros(p2-p,size(x,2))];
            downsample = @(x) x(1:p,:);
        end
        if exist('hadamard','file')==3
            % use my mex file
            H   = @(x) hadamard( x )/sqrt( p2 );
        else
            warning('kmeans_sketched:slowCode','using slow code; try to compile hadamard.c and put it into the path');
            H   = @(x) fwht( x, [], 'hadamard' )*sqrt(p2);
        end
        Ht  = H;
    elseif strcmpi( SketchType, 'DCT' )
        H   = @(x) dct(x); % no upsampling
        Ht  = @(x) idct(x); % transpose
    elseif strcmpi( SketchType, 'Nothing' ) || strcmpi( SketchType,'none' )
        H   = @(x) x;
        Ht  = H;
    else
        fprintf(2,'SketchType was %s\n', SketchType );
        error('bad type for "SketchType"');
    end
    
    if strcmpi( SketchType, 'Nothing' ) || strcmpi( SketchType,'none' )
        DiagRademacher   = @(x) x;
        % Note: if X has a lot of zero entries,
        %   then if we don't mix, after we sample, we'll have zero entries
        %   they will be mistaken in the updates. So, add a small offset
        X   = X + 2*eps; % eps is machine epsilon
    else
        % Nov 21 2015, found bug. Allow us to recreate bug so we can re-test results
        if FORCE_BUG 
            d   = sign(rand(p2,1));
        else
            d   = sign(randn(p2,1));
        end
        DD  = spdiags( d, 0, p2, p2 );
        DiagRademacher   = @(x) DD*x;
    end
    
    mix   = @(X) H(DiagRademacher(upsample(X))); % Preconditioning
    unmix = @(X) downsample( DiagRademacher( Ht(X) ) );
    
    if LoadFromDisk
        XFull   = [];
        X = sampleAndMixFromLargeFile( DataFile, SparsityLevel, mix, p2,...
            'ColumnSamples',ColumnSamples,'MB_limit',MB_limit,...
            'Verbose',DataFileVerbose);
    else
        if nargout > 4
            XFull = X; % save this for testing
        end
        X   = mix(X);
        
%         % small_p     = round( SparsityLevel*p );
%         small_p     = round( SparsityLevel*p2 );
        
        if SparsityIgnoreUpsampling
            small_p     = round( SparsityLevel*p );
        else
            small_p     = round( SparsityLevel*p2 );
        end
        
        Y           = spalloc(p,n,small_p*n);
        replace = false;
        for j = 1:n
            ind         = randsample(p2,small_p, replace );
            Y(ind,j)    = X(ind,j)/SparsityLevel;
        end
        X   = Y;
    end
    
    
    if strcmpi(Display,'iter') || strcmpi(Display,'final')
       fprintf('Randomly taking %.1f%% of the data; actual dataset is %.1f%% sparse\n',...
           100*SparsityLevel, 100*nnz(X)/numel(X) );
    end
    
    if MLcorrection
        % a bit of extra memory, but makes it easy
        NormalizationMatrix     = spones(X);
    end
else
    mix     = @(x) x; % do nothing
end


if ischar(start)
    switch lower(start)
        case 'uniform'
            mn  = full(min(X(:)));
            mx  = full(max(X(:)));
    end
end

bestObjective = Inf;
for nTrials = 1:Replicates
    
    if ischar(start)
        switch lower(start)
            case 'sample'
                ind         = randsample(n,K);
                centers     = full(X(:,ind));
            case 'uniform'
                centers     = (mx-mn)*rand(p2,K) - mn;
            case {'arthur','++','kmeans++','k-means++','k-means-++'}
                centers     = full(Arthur_initialization(X,K));
            otherwise
                error('cannot handle other types of "Start" values');
        end
    else
        % start is an array of the cluster centers, provided by the user
        if ~ColumnSamples
            start = start'; % want it of the size p x K
        end
        centers = mix( start );
        if Replicates > 1
            warning('kmeans_sparsified:deterministicCenters',...
                'initialization is specified, so running more than 1 replicate is not helpful');
        end
    end
    
    for its = 1:MaxIter
        
        % Find assignments
        [assignments,distances]     = findClusterAssignments(X,centers);
        
        % Update cluster centers
        centersOld  = centers;
        dropCenters = [];
        for ki = 1:K
            ind             = find( assignments == ki );
            if isempty(ind)
                warning('kmeans_sparsified:dropCluster','cluster has lost all its members');
                switch lower(EmptyAction)
                    case 'singleton'
                        [~,iMax] = max( distances );
                        centers(:,ki) = X(:,iMax);
                    case 'error'
                        error('One cluster lost all its members');
                    case 'drop'
                        dropCenters(end+1) = ki;
                    otherwise
                        error('invalid EmptyAction choice');
                end
                %centers(:,ki)   = 0;
            else
                if MLcorrection
                    centers(:,ki)   = SparsityLevel*full( sum(X(:,ind),2))./(full(sum(NormalizationMatrix(:,ind),2) )+1e-16);
                else
                    centers(:,ki)   = mean( full(X(:, ind ) ), 2 );
                end
            end
            
        end
        if ~isempty( dropCenters )
            centers = centers(:, setdiff(1:K,dropCenters) );
            centersOld = centersOld(:, setdiff(1:K,dropCenters) );
            assignments = []; % could recompute if we wanted to
            K       = size( centers, 2 );
        end
        
        dff     = norm( centersOld - centers,'fro');
        obj     = sqrt(sum(distances.^2));
        if strcmpi(Display,'iter') && ~mod(its,PrintEvery)
            fprintf('Iter: %3d; change in cluster centers: %.2e; objective: %.2e\n', ...
                its, dff, obj );
        end
        if dff < tol 
            break;
        end
        
        if any(isnan(centers))
%             disp('Found NaN: breaking for debugging purposes');
%             keyboard
            error('Found NaN in centers');
        end
        
    end
    
  
    if obj < bestObjective
        best            = true;
        objString       = ' (this is the best trial so far)';
        bestObjective = obj;
        bestAssignments = assignments;
        bestDistances   = distances;
        bestCenters     = centers;
    else
        best            = false;
        objString       = sprintf(' (best so far was %.2e)',bestObjective );
    end
    
    if strcmpi(Display,'iter') || (strcmpi(Display,'final')&& best )
        fprintf('Trial %3d of %3d total, objective %.2e%s\n',...
            nTrials, Replicates, obj, objString );
    end
end

SUMD    = zeros(K,1);
for ki = 1:K
    ind_i    =  bestAssignments == ki ;
    SUMD(ki) = sum(distances(:,ind_i).^2); % note that we do NOT take sqrt
end

if Sparsify
    bestCenters   = unmix( bestCenters ); % undo DCT or Hadamard if necessary
    
    if nargout > 4
        if LoadFromDisk
            % FIXME
            warning('Requires a second pass over the dataset');
            [assignments_twoPass,D_twoPass]     = ...
                recalculateAssignmentLargeFile(DataFile, bestCenters, ...
                'ColumnSamples',ColumnSamples,'MB_limit',MB_limit,...
                'Verbose',DataFileVerbose);
        else
            centers_twoPass = zeros(p,K);
            for ki = 1:K
                ind  = find( bestAssignments == ki );
                if ~isempty(ind)
                    centers_twoPass(:,ki)   = mean( full(XFull(:, ind ) ), 2 );
                end
            end

        end
    end
else
    % If we do not sparsify, then there is no distinction in # of passes...
    if nargout > 4
        centers_twoPass = bestCenters;
    end
end

if ~ColumnSamples
    % Be compatible with Matlab's "kmeans" function
    bestCenters   = bestCenters';
    bestDistances = bestDistances';
    bestAssignments = bestAssignments';
    SUMD          = SUMD';
    if nargout > 4
        centers_twoPass = centers_twoPass';
    end
end
