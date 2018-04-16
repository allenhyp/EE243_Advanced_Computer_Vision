clc;
clear all;
close all;

I = imread('./house.tif');
img = im2single(I(:,:,1));
J = fft2(img);
J_shifted = fftshift(J);
% figure; imshow(img); title('input image');
figure; 
subplot(1,2,1), imshow(log(max(abs(J_shifted), 1e-6)),[]), colormap(gca,jet); title('magnitude of the FFT of I');
subplot(1,2,2), imshow(angle(J)), colormap(gca,jet); title('phase of the FFT of I');