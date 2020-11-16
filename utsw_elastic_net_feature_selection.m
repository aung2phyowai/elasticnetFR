function [coef_matrix, metric_elasticnet] = utsw_elastic_net_feature_selection(X, Y, alpha, num_iteration, holdout_ratio)
% =========================================================================
%
% Shaode Yu (yushaodemia AT 163 DOT com)
% -------------------------------------------------------------------------
% elastic net based feature selection and weighting
%
% %  input parameters
%      X: the data set with n samples and p features ([n, p])
%      Y: the data labels ([n, 1]) in {0, 1}
%         Note, benign 0; malignant 1
%     alpha: the value to balance L1 and L2 penalty
%     num_iteration: the number of iteration
%     holdout_ratio: the ratio of hold out data for testing 
%
% %  output parameters
%      coef_matrix: the coefficient matrix
%      metric_elasticnet: the 7 parameters computed for 
%                             elastic net based performance evaluation
% -------------------------------------------------------------------------
% v01 05/28/2020
% v02 11/16/2020
% -------------------------------------------------------------------------

% (1) to check input parameters
if nargin < 5
    holdout_ratio = 0.2;
end

if nargin < 4
    num_iteration = 100;
end

if nargin < 3
    alpha = 0.75;
end

if nargin < 2
    fprintf('ERROR: insufficient input parameters ...\n');
    coef_matrix = [];
    metric_elasticnet = [];
    return;
end

% (2) to check whether input data is correct
if size(X, 1) ~= size(Y, 1)
    fprintf('ERROR: feature dimension not match ...\n');
    coef_matrix = [];
    metric_elasticnet = [];
    return;
end

uniLabel = unique(Y);
if 2 == size(uniLabel, 1)
    if (0 ~= min(uniLabel)) || (1 ~= max(uniLabel))
        fprintf('ERROR: wrong labels ...\n');
        coef_matrix = [];
        metric_elasticnet = [];
        return;
    end
elseif 1 == size(uniLabel, 1)
    if (0 ~= uniLabel) || (1 ~= uniLabel)
        fprintf('ERROR: wrong labels ...\n');
        coef_matrix = [];
        metric_elasticnet = [];
        return;
    end    
else
    fprintf('ERROR: wrong labels ...\n');
        coef_matrix = [];
        metric_elasticnet = [];
        return;    
end

% (3) to start offline elastic net based feature ranking and prediction
fprintf('... start elastic net based feature selection \n');
metric = zeros(num_iteration, 7);
coeffx = zeros(num_iteration, size(X,2)+1);
for ii = 1 : num_iteration    
    n = length(Y);
    c = cvpartition(n,'HoldOut',holdout_ratio);
    idxTrain = training(c,1);
    idxTest = ~idxTrain;
    XTrain = X(idxTrain,:);
    yTrain = Y(idxTrain);
    XTest = X(idxTest,:);
    yTest = Y(idxTest);
    
    [B,FitInfo] = lasso(XTrain,yTrain,'Alpha',alpha,'CV',10);    
    idxLambda1SE = FitInfo.Index1SE;
    coef = B(:,idxLambda1SE);
    coef0 = FitInfo.Intercept(idxLambda1SE);
    
    yhat = XTest*coef + coef0;
    yhat = double(yhat>0.5);
    metric(ii,:) = utsw_binary_classification_metrics(yTest, yhat);
    
    coeffx(ii,1) = coef0;
    coeffx(ii,2:end) = coef';
    fprintf('...... (%d)/(%d) \n', ii, num_iteration);
end

coef_matrix = coeffx;
metric_elasticnet = metric;
end

