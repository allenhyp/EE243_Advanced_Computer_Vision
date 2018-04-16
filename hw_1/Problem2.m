clc;
clear all;
close all;

img = imread('./Assigment-1/images/gonzalezwoods725.PNG');
img = rgb2gray(img);

M = 4;
N = 4;
B = zeros(M,N);
for k = 0:3
    if k == 0
        alpha = sqrt(1/M);
    else
        alpha = sqrt(2/M);
    end
    for n = 0:3
        B(n+1,k+1) = alpha*cos(k*pi*(2*n+1)/(2*M));
    end
end

figure;
k = 1;
for p = 1:4
    for q = 1:4
        dct_basis = B(:,p)*B(:,q)';
        subplot(4,4,k), imagesc(dct_basis); colormap(gray);
        k = k+1;
    end
end
