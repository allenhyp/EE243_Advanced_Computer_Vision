function [node_index nc_result] = NcutPartition(I, W, nc_th, a_th)
N = length(W);
d = sum(W, 2);
D = sparse(N,N);
for i = 1:N
    D(i,i) = d(i);
end

[Y,lambda] = eigs(D-W, D, 2, 'sm'); % (D - W)Y = lambda * D * Y
eig_vector_2 = Y(:, 2);

split_point = median(eig_vector_2);  % starting point for fminsearch
split_point = fminsearch('NcutValue', split_point, [],eig_vector_2, W, D);

partition_1 = find(eig_vector_2 > split_point);
partition_2 = find(eig_vector_2 <= split_point);

Ncut_value = NcutValue(split_point, eig_vector_2, W, D);
if (length(partition_1) < a_th || length(partition_2) < a_th || Ncut_value > nc_th)
    node_index{1}   = I;
    nc_result{1} = Ncut_value; 
    return;
end

%recursive partition
[node_index_1 Ncut_1]  = NcutPartition(I(partition_1), W(partition_1, partition_1), nc_th, a_th);
[node_index_2 Ncut_2] = NcutPartition(I(partition_2), W(partition_2, partition_2), nc_th, a_th);

node_index   = cat(2, node_index_1, node_index_2);
nc_result = cat(2, Ncut_1, Ncut_2);
end