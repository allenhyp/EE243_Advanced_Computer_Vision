close all;
image_left = imread('viprectification_deskLeft.png');
image_right = imread('viprectification_deskRight.png');
[m, n, c] = size(image_right);

% get SIFT features and match
[fm_left, dp_left] = getSIFTFeatures(image_left);
[fm_right, dp_right] = getSIFTFeatures(image_right);
matched_features = matchFeatures(dp_left', dp_right');
coord_left = fm_left(1:2, matched_features(:, 1))';
coord_right = fm_right(1:2, matched_features(:, 2))';

% homogeneous
homo_left = cart2hom(coord_left)';
homo_right = cart2hom(coord_right)';
x_left = homo_left(:, 40);
x_right = homo_right(:, 40);

for i = 1:100 
    h = randperm(41, 8);
    for j = 1:8
        P(i,:) = [homo_left(1,h(i))*homo_right(1,h(i));...
                  homo_left(2,h(i))*homo_right(1,h(i));...
                  homo_right(1,h(i));...
                  homo_left(1,h(i))*homo_right(2,h(i));...
                  homo_left(2,h(i))*homo_right(2,h(i));
                  homo_left(1,h(i));...
                  homo_left(1,h(i));...
                  homo_left(2,h(i));...
                  1];
    end
    [U, S, V] = svd(P);
    v_t = V(:, 9);
    F = reshape(v_t, 3, 3)';
    [UU, SS, VV] = svd(F);
    FF(:, :, i) = UU*diag([SS(1, 1) SS(2, 2) 0])*VV';
    error(j) = (x_left'*FF(:,:,i)*x_right)/((norm(FF(:,:,i)'*x_left)^2)+(norm(FF(:,:,i)*x_right)^2));
end

mean_error = mean(error);
[min_error, index] = min(error);
disp('minimum error =');
min_error
disp('fundamental matrix');
FF(:, :, index)

