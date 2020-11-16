# elastic net based feature ranking

## introduction

Elastic net is preferred in variable selection and widely used in intelligent diagnosis. However, its selected features are changed when the training set changes. 
To enhance the consistency of elastic net in feature selection, we propose an elastic net based feature ranking approach which considers both the frequency and weights 
of features.

## idea

The idea consists of three steps

(1) multiple times of elastic net based feature ranking and weight estimation

(2) feature importance ranking

(2.1) frequency based feature ranking

(2.2) weights based locally modification of ranks for features with equal frequency

(3) svm based comparison of incremental feature selection on breast cancer data sets

## Citation

If you think this is helpful, please cite

    Shaode Yu, Haobo Chen, Hang Yu, Zhicheng Zhang, Xiaokun Liang, Wenjian Qin, Yaoqin Xie, Ping Shi, "Elastic Net based Feature Ranking and Selection", under review.
