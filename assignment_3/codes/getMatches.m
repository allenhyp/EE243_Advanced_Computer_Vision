function matches = getMatches(featI,featR)

% featI and featR are two feature matrices of dim N1 x n_feat and N2 x n_feat respectively.
% matches is a N x 2 matrix indicating the indices of matches. N <= min(N1,N2)

matches = [];

for i = 1:size(featI,1)
%     [val,pos] = max(featI(i,1:16)*featR(:,1:16)' - mean(abs(repmat(featI(i,17:18),[size(featR,1),1])-featR(:,17:18)),2)')
    [val,pos] = max(featI(i,1:16)*featR(:,1:16)');
    if val > 0.65
        matches = [matches; [i, pos]];
    end
end
