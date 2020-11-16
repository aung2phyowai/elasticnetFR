function [fir_index, feature_frequency, feature_weights] = utsw_elastic_net_feature_importance_ranking(coef_matrix, ratio)
% =========================================================================
%
% Shaode Yu (yushaodemia AT 163 DOT com)
% -------------------------------------------------------------------------
% offline feature ranking based on the frequency and weights of
%                                  selected features via elastic net
%
% %  input parameters
%      coef_matrix: estimated coefficients matrix from 
%                     multiple times of elastic net based feature selection
%      ratio: the baseline ratio of frequency of selected features
%
% %  output parameters
%      fir_index: index of features from the most to the least important
%      feature_frequency: selection frequency of features via elastic net
%      feature_weights: the sum of estimated weights of each feature
% -------------------------------------------------------------------------
% v01 11/16/2020
% -------------------------------------------------------------------------

% (1) to check the parameters
if nargin < 2
    ratio = 0.49;
end

% (2) to extract the feature weights
coef = coef_matrix(:, 2:end);

% (3) to count the frequency of selected features
freCoef = sum( abs(coef)>0 ,1); % coef not equal to 0
% (3.1) frequency based feature ranking 
[fea_freqy, fea_index] = sort(freCoef,'descend');
% here we get the frequency, and corresponding index

% (3.2) to further consider feature weights when the frequencies are equal
coef_sum = sum(coef, 1); % sum of the coefficient weights
fea_weight = coef_sum(fea_index);
% (3.2.1) to modify the ranks according to their weights
uni_freqy = unique(fea_freqy);
uni_freqy = uni_freqy(end:-1:1);
uni_freqy = uni_freqy(uni_freqy> round(size(coef,1)*ratio));

for ii = 1 : size(uni_freqy,2)
    tmp_indx = find(fea_freqy == uni_freqy(ii));
    if size(tmp_indx,2) > 1
        tmp_weit = fea_weight(tmp_indx);
        [~, sort_weit] = sort(tmp_weit,'descend');
        
        fea_index(tmp_indx) = fea_index(tmp_indx(sort_weit));
        fea_weight(tmp_indx) = fea_weight(tmp_indx(sort_weit));
    end
    
end

fir_index = fea_index;
feature_frequency = fea_freqy;
feature_weights = fea_weight;
end

