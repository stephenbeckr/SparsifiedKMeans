%{
Example testing kmeans using Matlab's stat toolbox version,
 our own version, and our own version that uses the sparsification feature

Stephen.Becker@Colorado.edu, Aug 4 2015
%}

% Setup path, install mex files if necessary:
setup_kmeans

%% Generate synthetic data
rng(234);
k       = 5;
% p       = 100; n = 2e3;
p       = 512; n = 5e3; % Used this to make the .png image

centersTrue = randn(p,k);
X           = randn(p,n);
for ki = 1:k
    X(:, (ki-1)*n/k+1:ki*(n/k) )    = repmat(centersTrue(:,ki), 1,n/k) + ...
        .1*randn(p,n/k);
end
figure(1); clf; plot( centersTrue(1,:),centersTrue(2,:), 'o','markersize',12);
hold all; plot( X(1,:), X(2,:), '*' );
title('First two dimensions of the data');

%% various versions

nReplicates     = 20;

% -- Matlab's version. Do not run this for n>1e4
% Peculiarities about Matlab:
%   - for R2014a, I cannot run 'Start','Sample' beacuse I get an error
%       with how their code calls randsample.
%   - "dist" is really distance *squared*
%   - therefore objective should be sqrt(sum(sumd)), not norm(sumd)
%   - they expect samples to be rows, e.g., n x p matrix. Output
%       follows the same convention
tic
[indx,centers,sumd,dist] = kmeans( X', k ,'Start','uniform','Replicates',nReplicates);
time_Matlab     = toc;
obj_Matlab      = sqrt(sum(sumd));
centers_Matlab  = centers';
fprintf('stats toolbox version:\tobjective %.3e, time %.2e\n', obj_Matlab, time_Matlab );

% -- Our version, nothing fancy yet, just replicating same results as Matlab
%   but a smarter implementation
warning('off','kmeans_sparsified:dropCluster');
tic
[indx,centers,sumd,dist_sketch] = kmeans_sparsified( X, k,'ColumnSamples',true,...
    'Display','off','Replicates',nReplicates,'Sparsify',false, 'start','++');
time_faster     = toc;
obj_faster      = norm(dist_sketch);
centers_faster = centers;

fprintf('our version:\t\tobjective %.3e, time %.2e\n', obj_faster, time_faster );

% -- Our version, sampling the data to make it very fast
%   (Note: it will not necessarily be much faster until n>1e4 or so)
SparsityLevel = 0.05;
tic
[indx,centers,sumd,dist_sketch] = kmeans_sparsified( X, k,'ColumnSamples',true,...
    'Display','off','Replicates',nReplicates,...
    'Sparsify',true,'SparsityLevel',SparsityLevel );
time_fastest    = toc;
obj_fastest     = norm(dist_sketch);
centers_fastest = centers;
fprintf('our sparse version:\tobjective %.3e, time %.2e\n', obj_fastest, time_fastest );
%% Plot results
figure(1); clf;
str = {};
plot( X(1,:), X(2,:), 'k*' );  
  str{1}=sprintf('data samples (p=%d, n=%d)',p,n);
hold all; 
plot( centers_Matlab(1,:), centers_Matlab(2,:), 'ws','markersize',...
    17,'markerfacecolor','b' ,'linewidth',2 );
  str{end+1}=sprintf('K-means (stat toolbox), %.2f seconds',time_Matlab);
plot( centers_faster(1,:), centers_faster(2,:), 'w^','markersize',...
    12,'markerfacecolor','g' ,'linewidth',2 );
  str{end+1}=sprintf('K-means (our toolbox), %.2f seconds',time_faster);
plot( centers_fastest(1,:), centers_fastest(2,:), ...
    'wd','markersize',12,'markerfacecolor','r','linewidth',2 );
  str{end+1}=sprintf('Sparsified K-means (%.0f%% sparse), %.2f seconds',...
      100*SparsityLevel,time_fastest);

set(gca,'fontsize',12);
legend(str);
xlabel('dimension 1');
ylabel('dimension 2');
title('First two dimensions of the data');