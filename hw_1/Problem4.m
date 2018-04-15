clear all
close all

img = imread('./Assigment-1/images/lena_gray_256_noisy.png');
J = dct2(img);
cutoff = round(0.5*256);
high_f = fliplr(tril(fliplr(J), cutoff));
low_f = J - high_f;

high_o = idct2(high_f);
low_o = idct2(low_f);
figure; imshowpair(high_o, low_o, 'montage');
