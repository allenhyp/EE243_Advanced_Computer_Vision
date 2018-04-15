clc;
clear all;
close all;

fft_obj = vision.FFT;
I = imread('./Assigment-1/TEST_IMAGES/house.tif');
img = im2single(I(:,:,1));
J = step(fft_obj, img);
J_shifted = fftshift(J);
figure; imshow(img); title('input image');
figure; imshow(log(max(abs(J_shifted), 1e-6)),[]), colormap(jet); title('magnitude of the FFT of I');
