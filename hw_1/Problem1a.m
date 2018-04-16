clc;
clear all;
close all;

I = imread('./Assigment-1/TEST_IMAGES/house.tif');
img = im2single(I(:,:,1));
J = fft2(img);
J_shifted = fftshift(J);
figure; imshow(img); title('input image');
figure; imshow(log(max(abs(J_shifted), 1e-6)),[]), colormap(gca,jet); colorbar; title('magnitude of the FFT of I');
figure; imshow(angle(J)), colormap(gca,jet); title('phase of the FFT of I');