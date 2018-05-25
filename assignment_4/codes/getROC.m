function [TPR,FPR] = getROC(pred,gt)

% gt is the ground truth vector of 1 or 0 of size n_samples x 1. 1
% indicates a positive and 0 negative
% pred is a vector of predictions of size n_samples x 1
% TPR is the True Positive Rate
% FPR is the False Positive Rate

% FILL IN
% n = length(pred);
% TPR = zeros(n,1);
% FPR = zeros(n,1);
% tp = 0;
% fp = 0;
% for i = 1:length(pred)
%     if pred(i)==1
%         if gt==1
%             tp = tp+1;
%         else
%             fp = fp+1;
%         end
%     end
%     m = max(1, tp+fp);
%     TPR(i) = tp/(m);
%     FPR(i) = fp/(m);
[FPR, TPR] = perfcurve(gt, pred, 1);
end


    
