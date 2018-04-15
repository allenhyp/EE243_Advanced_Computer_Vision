clc;
clear all;
close all;

img = imread('./Assigment-1/images/gonzalezwoods725.PNG');
img = rgb2gray(img);
fft_obj = vision.FFT;
% ifft_obj = vision.IFFT;
[m,n] = size(img);
d = int16(m/4);
k = 1;
figure;
for i = 0:3
    for j = 0:3
        seg_img = single(img(1+i*d:1+i*d+d,1+j*d:1+j*d+d));
        J = step(fft_obj, seg_img);
        J_shifted = fftshift(J);
        result_img = log(max(abs(J_shifted), 1e-6));
        subplot(4,4,k), imshow(result_img); colormap(gca,jet(64));
        k = k+1;
    end
end

figure;
k = 1;
for i = 0:3
    for j = 0:3
        seg_img = single(img(1+i*d:1+i*d+d,1+j*d:1+j*d+d));
        J = dct2(seg_img);
        result_img = log(abs(J));
        subplot(4,4,k), imshow(result_img,[]); colormap(gca,jet(64));
        k = k+1;
    end
end