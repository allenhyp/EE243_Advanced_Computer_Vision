clc;
clear all;
close all;

I = imread('./Assigment-1/TEST_IMAGES/house.tif');
img = im2single(I(:,:,1));
img_f = imgaussfilt(img,3);
[m,n] = size(img_f);
resample_rate = 16;
img_resample = img_f(1:resample_rate:m,1:resample_rate:n);
img_r_o = imresize(img_resample, [m,n]);

figure; imshowpair(img, img_r_o, 'montage');
error_rate = 0;
for i = 1:m
    for j = 1:n
        error_rate = error_rate + img(i,j) - img_r_o(i,j);
    end
end
error_rate = abs(error_rate)/256