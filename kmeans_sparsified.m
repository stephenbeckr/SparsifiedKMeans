function [bestAssignments, bestCenters, SUMD, bestDistances, bestDistancesTrue] = kmeans_sparsified(X, K, varargin)
% [IDX, C, SUMD, D] = kmeans_sparsified(X, K)
%
% Note: to maintain compatability with Matlab, we assume the points
%   are ROWS of X, so X is a n x p matrix
% If you want columns to be the points, then set 'ColumnSamples',true
%
% IMPORTANT: to run the "fast" (sparsified) code, make sure to 
%   set 'Sparsify',true
%
% Examples:
%   [IDX,C] = kmeans_sparsified( X, 5 )
%   [IDX,C] = kmeans_sparsified( X', 5,'ColumnSamples',true,'tol',1e-4)
%
% Most of the options are the same as Matlab, though we do not yet
%   have SUMD implemented
% Options:
%   'Replicates', r         Run K-Means r times, keeping the best
%   'Start'                 if 'sample', picks K random data points
%                           if 'uniform', picks K points uniformly from
%                               the space
%                           if 'Arthur' or '++', uses Arthur/Vassilvitskii 2007
%                               K-means++ algorithm to initialize
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

% Stephen Becker and Farhad Pourkamali-Anaraki
% Stephen.Becker@Colorado.edu, 8/5/2015


p = inputParser;
addParameter(p,'Replicates',1);
expectedStart = {'sample','uniform','Arthur','k-means++','++'};
addParameter(p,'Start','Arthur',@(x) any(validatestring(x,expectedStart)));
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
parse(p,varargin{:});

Replicates  = p.Results.Replicates;
start       = p.Results.Start;
MaxIter     = p.Results.MaxIter;
Display     = p.Results.Display;
tol         = p.Results.Tol;
Sketch      = p.Results.Sparsify;
SparsityLevel    = p.Results.SparsityLevel; % used when Sketching
SketchType  = p.Results.SketchType; % Hadamard or DCT or Nothing
EmptyAction = p.Results.EmptyAction;
PrintEvery  = p.Results.PrintEvery;
ColumnSamples = p.Results.ColumnSamples;
MLcorrection= p.Results.MLcorrection && Sketch;

if ~ColumnSamples
    X   = X';
end

[p,n]       = size(X);
if n<K
    error('kmeans_sparsified:badDimensions','X must have more samples than the number of clusters.');
end

p2          = p;
if Sketch
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
        D   = @(x) x;
        % Note: if X has a lot of zero entries,
        %   then if we don't mix, after we sample, we'll have zero entries
        %   they will be mistaken in the updates. So, add a small offset
        X   = X + 2*eps; % eps is machine epsilon
    else
        d   = sign(rand(p2,1));
        DD  = spdiags( d, 0, p2, p2 );
        D   = @(x) DD*x;
    end
    
    if nargout > 4
        XFull = X; % save this for testing
    end
    X   = H(D(upsample(X))); % Mixing
    
%     small_p     = round( SparsityLevel*p );
    small_p     = round( SparsityLevel*p2 );
    Y           = spalloc(p,n,small_p*n);
    for j = 1:n
        replace = false;
        ind     = randsample(p2,small_p, replace );
        Y(ind,j)    = X(ind,j)/SparsityLevel;
    end
    X   = Y;
    
    if strcmpi(Display,'iter') || strcmpi(Display,'final')
       fprintf('Randomly taking %.1f%% of the data; actual dataset is %.1f%% sparse\n',...
           100*SparsityLevel, 100*nnz(X)/numel(X) );
    end
    
    if MLcorrection
        % a bit of extra memory, but makes it easy
        NormalizationMatrix     = spones(X);
    end
    
end



switch lower(start)
    case 'uniform'
        mn  = full(min(X(:)));
        mx  = full(max(X(:)));
end

bestObjective = Inf;
for nTrials = 1:Replicates
    
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
                    % Not sure why we need SparsityLevel in there...
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

SUMD = []; % not used

if Sketch
    bestCenters   = downsample( D( Ht(bestCenters) ) ); % undo mixing
    
    
    if nargout > 4
        [bestAssignments,bestDistancesTrue]     = findClusterAssignments(XFull,bestCenters);
    end
else
    if nargout > 4, bestDistancesTrue = bestDistances; end
end

if ~ColumnSamples
    % Be compatible with Matlab's "kmeans" function
    bestCenters   = bestCenters';
    bestDistances = bestDistances';
    bestAssignments = bestAssignments';
    if nargout > 4
        bestDistancesTrue = bestDistancesTrue';
    end
end
