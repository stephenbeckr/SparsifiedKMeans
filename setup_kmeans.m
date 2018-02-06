function setup_kmeans
% setup_kmeans
%
%   This adds the appropriate paths needed for the kmeans package
%   It also compiles mex files if they have not already been compiled
%
% Stephen.Becker@Colorado.edu  8/4/2015

%% Setup path

baseDirectory = fileparts(mfilename('fullpath'));
addpath(genpath_ignoreHiddenDir(baseDirectory));

%% Compile the KMeans sparse-matrix-assignment code
if exist('SparseMatrixMinusCluster','file')~=3
    here = pwd;
    cd( fullfile(baseDirectory,'private') );
    disp('Compiling SparseMatrixMinusCluster code');
    mex -largeArrayDims -O SparseMatrixMinusCluster.c
    cd(here);
end
if exist('SparseMatrixInnerProduct','file')~=3
    here = pwd;
    cd( fullfile(baseDirectory,'private') );
    disp('Compiling SparseMatrixInnerProduct code');
    mex -largeArrayDims -O SparseMatrixInnerProduct.c
    cd(here);
end
if exist('SparseMatrixColumnNormSq','file')~=3
    here = pwd;
    cd( fullfile(baseDirectory,'private') );
    disp('Compiling SparseMatrixColumnNormSq code');
    mex -largeArrayDims -O SparseMatrixColumnNormSq.c
    cd(here);
end

%% FJLT (Fast Hadamard) code
% Both normal and parallel (pthread) versions
if exist('hadamard','file')~=3
    here = pwd;
    cd( fullfile(baseDirectory,'private') );
    disp('Compiling fast Hadamard code');
    % Not all versions of Matlab support "maxNumCompThreads" (Octave doesn't either):
    if 2 == exist('maxNumCompThreads','file')
        threads=maxNumCompThreads(); % good value of NTHREADS
    else
        threads = 1; % you can change this by hand if you like
    end
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


function p = genpath_ignoreHiddenDir(d)
%%
% initialise variables
classsep = '@';  % qualifier for overloaded class directories
packagesep = '+';  % qualifier for overloaded package directories
p = '';           % path to be returned

% Generate path based on given root directory
files = dir(d);
if isempty(files)
  return
end

% Add d to the path even if it is empty.
p = [p d pathsep];

% set logical vector for subdirectory entries in d
isdir = logical(cat(1,files.isdir));
%
% Recursively descend through directories which are neither
% private nor "class" directories.
%
dirs = files(isdir); % select only directory entries from the current listing

for i=1:length(dirs)
   dirname = dirs(i).name;
   if    ~strcmp( dirname,'.')          && ...
         ~strcmp( dirname,'..')         && ...
         ~strncmp( dirname,classsep,1) && ...
         ~strncmp( dirname,packagesep,1) && ...
         ~strcmp( dirname,'private') && ...
         ~strcmpi( dirname(1), '.' ) % added in order to exclude .git/ files
      p = [p genpath(fullfile(d,dirname))]; % recursive calling of this function.
   end
end
