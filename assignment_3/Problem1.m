close all
clear all

I = imread('house.tif');
I=imresize(I(:,:,1),[100 100]);
% I=I(:,:,1);
[rows, cols, c] = size(I);
N = rows * cols;

r = 2;
sig_i = 4;
sig_x = 6;
nc_threshold = 0.03;
area_threshold = 100;

V = zeros(N,c);
W = sparse(N,N); 
X_t = zeros(rows, cols, 2);
X = zeros(N,1,2);
F = zeros(N,1,c);


for k = 1:c
    cnt = 1;
    for i = 1:cols
      for j = 1:rows
          V(cnt,k) = I(j,i,k);
          cnt = cnt + 1;        
      end
    end 
end

for i = 1:rows
    for j = 1:cols
        X_t(i,j,1) = i;
        X_t(i,j,2) = j;
    end
end

for k = 1:2
      cnt = 1;
      for i = 1:cols
        for j = 1:rows
            X(cnt,1,k) = X_t(j,i,k);
            cnt = cnt + 1;        
        end
      end 
end

for k = 1:c
   cnt = 1;
   for i = 1:cols
       for j = 1:rows
           F(cnt,1,k) = I(j,i,k);     
           cnt = cnt + 1;        
       end
   end 
end
F = uint8(F); %uint class required for addition compatibility with spatial

r_t = floor(r);
for m =1:cols
    for n =1:rows
        
        %satisfies X(j)-r < X(i) < X(j)+r  
        range_cols = (m - r_t) : (m + r_t); 
        range_rows = ((n - r_t) :(n + r_t))';
        valid_col_index = range_cols >= 1 & range_cols <= cols;  %valid col. index
        valid_row_index = range_rows >= 1 & range_rows <= rows;  %valid row index
        
        range_cols = range_cols(valid_col_index);   %range of cols. and rows satisfying euclidean distance metric  
        range_rows = range_rows(valid_row_index);
        
        %current_vertex index
        cur_vertex = n + (m - 1) * rows;
 %-----------------------------------------------------------------------------------------%       
        
        l_r = length(range_rows);
        l_c = length(range_cols);
        temp_1 = zeros(l_r,l_c);
        temp_2 = zeros(l_r,l_c);
        for i = 1:length(range_rows)
            for j = 1:length(range_cols)
                temp_1(i,j) = range_rows(i,1);
            end
        end
                   
        for i = 1:length(range_rows)
            for j = 1:length(range_cols)
                temp_2(i,j) = ((range_cols(1,j) -1) .*rows);
            end
        end
        n_vertex_temp = temp_1 + temp_2;
        n_vertex = zeros(l_r*l_c,1);
        cnt = 1;
        for i = 1:l_c
            for j = 1:l_r
                n_vertex(cnt,1) = n_vertex_temp(j,i);
                cnt = cnt + 1;        
            end
        end 
        
        X_J = zeros(length(n_vertex),1,2); 
        for k = 1:2
            for i = 1:length(n_vertex)
                X_J(i,1,k) = X(n_vertex(i,1),1, k);
            end
        end      
                
        
        X_I_temp = X(cur_vertex, 1, :);
        X_I = zeros(length(n_vertex),1,2);  
      
        for i = 1:length(n_vertex)
            for k = 1:2
                X_I(i,1,k) = X_I_temp(1,1,k);
            end
        end
        diff_X = X_I - X_J;
        diff_X = sum(diff_X .* diff_X, 3); % squared euclid distance
        
        % |X(i) - X(j)| <= r 
        valid_index = (sqrt(diff_X) <= r);
        n_vertex = n_vertex(valid_index);
        diff_X = diff_X(valid_index);

        % feature vector disimilarity
        F_J = zeros(length(n_vertex),1,c); 
        for i = 1:length(n_vertex)
            for k = 1:c
                a = n_vertex(i,1);
                F_J(i,1,k) = F(a,1,k);
            end
        end
        F_J = uint8(F_J);
        
        FI_temp = F(cur_vertex, 1, :);
        F_I = zeros(length(n_vertex),1,c);  
        for i = 1:length(n_vertex)
            for k = 1:c
                F_I(i,1,k) = FI_temp(1,1,k);
            end
        end
        F_I = uint8(F_I);        
        
        diff_F = F_I - F_J;
        diff_F = sum(diff_F .* diff_F, 3); 
        W(cur_vertex, n_vertex) = exp(-diff_F / (sig_i*sig_i)) .* exp(-diff_X / (sig_x*sig_x)); % for squared distance
        
    end
end

% call to partition routine
node_index = (1:N)'; 
[node_index Ncut] = NcutPartition(node_index, W, nc_threshold, area_threshold);


%  node_indexes to images

for i=1:length(node_index)
    seg_I_temp_1 = zeros(N, c);
    seg_I_temp_1(node_index{i}, :) = V(node_index{i}, :);
    seg_I_temp_2{i} = (reshape(seg_I_temp_1, rows, cols, c));
    seg_I{i} = uint8(seg_I_temp_2{i});
    
end

figure
seg_length = length(seg_I);
for i=1:seg_length
    subplot(seg_length/3,3,i)
    imshow(seg_I{i});
    %imwrite(Segment_I{i}, sprintf('test%d.jpg', i));
    %fprintf('Ncut(%d) = %f\n', i, Ncut{i});
end

