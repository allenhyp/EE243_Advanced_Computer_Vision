I = imread('./house.tif');
I = imresize(I(:,:,1), [100 100]);
[rows, cols, c] = size(I);
N = rows * cols;
% Parameters init
r = 2;
sig_i = 3;
sig_x = 5;
nc_threshold = 0.15;
area_threshold = 100;
V = zeros(N, c); % vertex of the graph
W = sparse(N, N); % similarity matrix
X_t = zeros(rows, cols, 2);
X = zeros(N, 1, 2); % spatial location matrix
F = zeros(N, 1, c); % Intensity feature vectors

for i = 1:rows
    for j = 1:cols
        X_t(i, j, 1) = i;
        X_t(i, j, 2) = j;
    end
end

for k = 1:c
    cnt = 1;
    for i = 1:cols
        for j = 1:rows
            V(cnt, k) = I(j, i, k);
            F(cnt, 1, k) = I(j, i, k);
            if k < 3
                X(cnt, 1, k) = X_t(j, i, k);
            end
            cnt = cnt + 1;
        end
    end
end
F = uint8(F);

r_1 = floor(r);
for m = 1:cols
    for n = 1:rows
        range_cols = (m-r_1):(m+r_1);
        range_rows = ((n-r_1):(n+r_1))';
        valid_col_index = range_cols >= 1 & range_cols <= cols;
        valid_row_index = range_rows >= 1 & range_rows <= rows;
        range_cols = range_cols(valid_col_index);
        range_rows = range_rows(valid_row_index);
        cur_vertex = n+(m-1)*rows;
        l_c = length(range_cols);
        l_r = length(range_rows);
        temp_1 = zeros(l_r, l_c);
        temp_2 = zeros(l_r, l_c);
        for i = 1:length(range_rows)
            for j = 1:length(range_cols)
                temp_1(i, j) = range_rows(i, 1);
                temp_2(i, j) = ((range_cols(1,j)-1).*rows);
            end
        end
        n_vertex_temp = temp_1+temp_2;
        n_vertex = zeros(l_r*l_c, 1);
        cnt = 1;
        for i = 1:l_c
            for j = 1:l_r
                n_vertex(cnt, 1) = n_vertex_temp(j, i);
                cnt = cnt + 1;
            end
        end
        X_J = zeros(length(n_vertex), 1, 2);
        X_I_temp = X(cur_vertex, 1, :);
        X_I = zeros(length(cur_vertex), 1, 2);
        for i = 1:2
            for j = 1:length(n_vertex)
                X_J(j, 1, i) = X(n_vertex(i, 1), 1, k);
                X_I(j, 1, i) = X_I_temp(1, 1, i);
            end
        end
        diff_X = X_I-X_J;
        diff_X = sum(diff_X.*diff_X, 3);
        valid_index = (sqrt(diff_X) <= r);
        n_vertex = n_vertex(valid_index);
        diff_X = diff_X(valid_index);

        F_J = zeros(length(n_vertex), 1, c);
        F_I_temp = F(cur_vertex, 1, :);
        F_I = zeros(length(n_vertex), 1, c);
        for i = 1:length(n_vertex)
            for j = 1:c
                t = n_vertex(i, 1);
                F_J(i, 1, j) = F(t, 1, j);
                F_I(i, 1, j) = F_I_temp(1, 1, j);
            end
        end
        F_J = uint8(F_J);
        F_I = uint8(F_I);
        diff_F = F_I-F_J;
        diff_F = sum(diff_F.*diff_F, 3);
        W(cur_vertex, n_vertex) = exp(-diff_F/(sig_i^2)).*exp(-diff_X/(sig_x^2));
    end
end

node_index = (1:N)';
[node_index, nc_result] = n_cut_partition(I, W, nc_threshold, area_threshold);

for i = 1:length(node_index)
    seg_I_temp_1 = zeros(N, c);
    seg_I_temp_1(node_index{i}, :) = V(node_index{i}, :);
%     seg_I_temp_2 = zeros(rows, cols, c);
    seg_I_temp_2{i} = (reshape(seg_I_temp_1, rows, cols, c));
    seg_I{i} = uint8(seg_I_temp_2{i});
end

for i = 1:length(seg_I)
    figure;
    imshow(seg_I{i});
    fprintf('normalized_cut(%d) = %f\n', i, nc_result{i});
end

function [node_index, nc_result] = n_cut_partition(I, W, nc_th, a_th)
    N = length(W);
    d = sum(W, 2);
    D = spdiags(d, 0, N, N);
%     for i = 1:N
%         D(i, i) = d(i);
%     end
    [y, lambda] = eigs(D-W, D, 2, 'sm'); % (D-W)y = labmda*D*y
    eigen_vector_2 = y(:, 2);
    split_point = median(eigen_vector_2);
    split_point = fminsearch('n_cut_value', split_point, eigen_vector_2, W, D);
    partition_1 = find(eigen_vector_2 > split_point);
    partition_2 = find(eigen_vector_2 <= split_point);
    nc_value = n_cut_value(split_point, eigen_vector_2, W, D);
    if(length(partition_1) < a_th || length(partition_2) < a_th || nc_value > nc_th)
        node_index{1} = I;
        nc_result{1} = nc_value;
        return;
    end
    
    [node_index_1, nc_result_1] = n_cut_partition(I(partition_1), W(partition_1, partition_1), nc_th, a_th);
    [node_index_2, nc_result_2] = n_cut_partition(I(partition_2), W(partition_2, partition_2), nc_th, a_th);
    node_index = cat(2, node_index_1, node_index_2);
    nc_result = cat(2, nc_result_1, nc_result_2);
end

function value = n_cut_value(sp, ev_2, W, D)
    x = (ev_2 > sp);
    x = (2*x)-1;
    d = sum(W, 2);
    k = sum(d(x > 0))/sum(d);
    b = k/(1-k);
    y = (1+x)-b*(1-x);
    value = (y'*(D-W)*y)/(y'*D*y);
end