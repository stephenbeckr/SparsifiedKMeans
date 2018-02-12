%{
This file demonstrates how to use the load-from-disk feature
 This feature is useful when the datafile is so large, like 10 TB,
 that you cannot load it all into RAM at once.

%}

% Setup path, install mex files if necessary:
setup_kmeans

%% Generate synthetic data
rng(234);
k       = 5;
p       = 500; n = 5e3; % Used this to make the .png image

centersTrue = randn(p,k);
X           = randn(p,n);
for ki = 1:k
    X(:, (ki-1)*n/k+1:ki*(n/k) )    = repmat(centersTrue(:,ki), 1,n/k) + ...
        .1*randn(p,n/k);
end
figure(1); clf; plot( centersTrue(1,:),centersTrue(2,:), 'o','markersize',12);
hold all; plot( X(1,:), X(2,:), '*' );
title('First two dimensions of the data');

%% Save the data to disk
% The code expects a .mat file with one variable in it (any variable name works)
myfile = fullfile(tempdir,'sampleInput.mat');
save( myfile, 'X', '-v7.3' ); % about 20 MB

%% Run standard k-means (in-core)
disp('--- Standard k-means ---');

nReplicates     = 20;

warning('off','kmeans_sparsified:dropCluster');
tic
[indx,centers,sumd,dist_sketch] = kmeans_sparsified( X, k,'ColumnSamples',true,...
    'Display','off','Replicates',nReplicates,'Sparsify',false, 'start','++');
time_faster     = toc;
obj_faster      = norm(dist_sketch);
centers_faster = centers;

fprintf('our version:\t\tobjective %.3e, time %.2e\n', obj_faster, time_faster );

%% -- Our version, sampling the data to make it very fast. Still in-core
%   (Note: it will not necessarily be much faster until n>1e4 or so)
disp('--- Sparsified k-means ---');
SparsityLevel = 0.05;
tic
[indx,centers,sumd,dist_sketch] = kmeans_sparsified( X, k,'ColumnSamples',true,...
    'Display','off','Replicates',nReplicates,...
    'Sparsify',true,'SparsityLevel',SparsityLevel );
time_fastest    = toc;
obj_fastest     = norm(dist_sketch);
centers_fastest = centers;
fprintf('our sparse version:\tobjective %.3e, time %.2e\n', obj_fastest, time_fastest );

%% New feature: loading from disk
disp('--- Sparsified k-means, on out-of-core data ---');
MB_limit    = 1; % size, in MB, of how much RAM we have to work with
                 % Try 500 or 1000 (e.g., 1 GB). Here, we set it very small
                 % just for this example.
tic
[indx,centers,sumd,dist_sketch] = kmeans_sparsified( [], k,'ColumnSamples',true,...
    'Display','off','Replicates',nReplicates,...
    'Sparsify',true,'SparsityLevel',SparsityLevel,'DataFile',myfile,...
    'MB_limit',MB_limit,'DataFileVerbose',true);
time_fastest_disk    = toc;
obj_fastest_disk     = norm(dist_sketch);
centers_fastest_disk = centers;
fprintf('our sparse version (from disk):\tobjective %.3e, time %.2e\n', obj_fastest_disk, time_fastest_disk );

%% Plot results
figure(1); clf;
str = {};
plot( X(1,:), X(2,:), 'k*' );  
  str{1}=sprintf('data samples (p=%d, n=%d)',p,n);
hold all; 
% plot( centers_Matlab(1,:), centers_Matlab(2,:), 'ws','markersize',...
%     17,'markerfacecolor','b' ,'linewidth',2 );
%   str{end+1}=sprintf('K-means (stat toolbox), %.2f seconds',time_Matlab);
plot( centers_faster(1,:), centers_faster(2,:), 'w^','markersize',...
    12,'markerfacecolor','g' ,'linewidth',2 );
  str{end+1}=sprintf('K-means (our toolbox), %.2f seconds',time_faster);
plot( centers_fastest(1,:), centers_fastest(2,:), ...
    'wd','markersize',12,'markerfacecolor','r','linewidth',2 );
  str{end+1}=sprintf('Sparsified K-means (%.0f%% sparse), %.2f seconds',...
      100*SparsityLevel,time_fastest);
plot( centers_fastest_disk(1,:), centers_fastest_disk(2,:), 'ws',...
    'markersize',12,'markerfacecolor','b','linewidth',2 );
  str{end+1}=sprintf('Sparsified K-means, reading from big .mat file (%.0f%% sparse), %.2f seconds',...
      100*SparsityLevel,time_fastest_disk);
  
set(gca,'fontsize',12);
legend(str);
xlabel('dimension 1');
ylabel('dimension 2');
title('First two dimensions of the data');