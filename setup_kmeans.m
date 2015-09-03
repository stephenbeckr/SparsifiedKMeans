function setup_kmeans
% setup_kmeans
%
%   This adds the appropriate paths needed for the kmeans package
%   It also compiles mex files if they have not already been compiled
%
% Stephen.Becker@Colorado.edu  8/4/2015

%% Setup path

baseDirectory = fileparts(mfilename('fullpath'));
addpath(genpath(baseDirectory));

%% Compile the KMeans sparse-matrix-assignmenet code
if exist('SparseMatrixMinusCluster','file')~=3
    here = pwd;
    cd( fullfile(baseDirectory,'private') );
    disp('Compiling SparseMatrixMinusCluster code');
    mex -largeArrayDims SparseMatrixMinusCluster.c
    cd(here);
end

%% FJLT (Fast Hadamard) code
% Both normal and parallel (pthread) versions
if exist('hadamard','file')~=3
    here = pwd;
    cd( fullfile(baseDirectory,'private') );
    disp('Compiling fast Hadamard code');
    threads=maxNumCompThreads(); % good value of NTHREADS
    threaddFlg = sprintf('-DNTHREADS=%d',threads);
    if isunix
        % Assuming we are using gcc, so I know some fancier flags
        % This might make a difference on new computers (> 2012) that have AVX
        mex -O CFLAGS="\$CFLAGS -march=native -O3" hadamard.c -DNO_UCHAR
        
        mex('-O',threaddFlg,'-UDEBUG','-DNO_UCHAR',...
            'CFLAGS="\$CFLAGS -pthread -O6"', ...
            'hadamard_pthreads.c');
    else
        mex -O hadamard.c
        mex('-O',threaddFlg,'-UDEBUG','-DNO_UCHAR','hadamard_pthreads.c');
    end
    cd(here);
end
