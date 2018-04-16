clc;
clear all;
close all;

img = imread('./gonzalezwoods725.PNG');
img = rgb2gray(img);
[a,b] = size(img);
M = 4;
B = zeros(M,M);
F = zeros(M,M);
for k = 0:3
    if k == 0
        alpha = sqrt(1/M);
    else
        alpha = sqrt(2/M);
    end
    for n = 0:3
        B(n+1,k+1) = alpha*cos(k*pi*(2*n+1)/(2*M));
        F(n+1,k+1) = exp(-1i*2*pi*k*n/M);
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

figure;
k = 1;
for u = 1:4
    for v = 1:4
        dft_basis = F(:,u)*F(:,v)';
        dft_basis_r = real(dft_basis)-imag(dft_basis);
        subplot(4,4,k), imagesc(dft_basis_r); colormap(gray);
        k = k+1;
    end
end

J = fft2(img);
J_shifted = fftshift(J);
figure; imshow(log(max(abs(J_shifted), 1e-6)),[]), colormap(gca,jet); title('magnitude of the FFT');
figure; imshow(angle(J)), colormap(gca,jet); title('phase of the FFT');

L = dct2(img);
figure; imshow(L), colormap(gca,jet); title('magnitude of the DCT');
