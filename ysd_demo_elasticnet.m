% Shaode Yu
% 08/15/2020
clear; close all; clc;

% (1) to prepare the data set
load uci_wdbc;
X = [ben; mal];
Y = [zeros(size(ben,1),1); ones(size(mal,1),1)];

% (2) elastic net based feature selection, parameter estimation, 
%          and classification
alpha = 0.75;
num_iteration = 50;
holdout_ratio = 0.2;
[coef_matrix, metric_elasticnet] = utsw_elastic_net_feature_selection(X, Y, alpha, num_iteration, holdout_ratio);

% (3) frequency and weights based feature importance ranking
frequency_baseline = 0.49;
[fir_index, feat_freq, feat_weights] = utsw_elastic_net_feature_importance_ranking(coef_matrix, frequency_baseline);
