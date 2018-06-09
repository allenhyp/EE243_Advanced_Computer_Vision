clear all; close all;

image_1 = imread('./images/Image1.tif');
image_2 = imread('./images/Image9.tif');
[corners_1, board_size_1] = detectCheckerboardPoints(image_1);
[corners_2, board_size_2] = detectCheckerboardPoints(image_2);

% figure;
% subplot(1, 2, 1);
% imshow(image_1);
% hold on;
% plot(corners_1(:,1),corners_1(:,2),'go');
% hold off;
% subplot(1, 2, 2);
% imshow(image_2);
% hold on;
% plot(corners_2(:,1),corners_2(:,2),'go');
% hold off;

[n, m] = size(corners_1);
P = zeros(2*n, 9);
P(1:2:2*n-1, 1:2) = corners_1;
P(1:2:2*n-1, 3) = 1;
P(2:2:2*n, 4:5) = -corners_1;
P(2:2:2*n, 6) = -1;
P(1:2:2*n-1, 7) = corners_2(:,2)'.*corners_1(:,1)';
P(2:2:2*n, 7) = -corners_2(:,1)'.*corners_1(:,1)';
P(1:2:2*n-1, 8) = corners_2(:,2)'.*corners_1(:,2)';
P(2:2:2*n, 8) = -corners_2(:,1)'.*corners_1(:,2)';
P(1:2:2*n-1, 9) = corners_2(:, 2)';
P(2:2:2*n, 9) = -corners_2(:, 1)';

[U, S, V] = svd(P);
sigmas = diag(S);
if length(sigmas) >= 9
    min_sigma = min(sigmas);
    [min_sigma_row, min_sigma_col] = find(S==min_sigma);
    q = double(vpa(V(:, min_sigma_col)));
else
    q = double(vpa(V(:,9)));
end

H = reshape(q, [3, 3])'