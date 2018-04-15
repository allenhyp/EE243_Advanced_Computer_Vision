clc;
clear all;
close all;

fft_obj = vision.FFT;
ifft_obj = vision.IFFT;
I = imread('./Assigment-1/TEST_IMAGES/house.tif');
img = im2single(I(:,:,1));
h = fspecial('gaussian',2,1);

img_b = imfilter(img,h);
img_t = imresize(img_b, 0.2);
J = step(fft_obj, img_t);
inverse = step(ifft_obj, J);
inverse = imresize(inverse, 1/0.2);
J_shifted = fftshift(J);
%     figure; imshow(img_b); title('input image');
result_img = log(max(abs(J_shifted), 1e-6));

figure; imshow(result_img), colormap(jet(64)); title('magnitude of the DFT of I');
figure; imshowpair(img, inverse, 'montage'); title('reconstruction by DFT (sample rate = 0.2)');
