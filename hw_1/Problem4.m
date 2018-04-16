clear all
close all

img = imread('./lena_gray_256_noisy.png');
J = dct2(img);
high_f = fliplr(tril(fliplr(J), round(0.4*256)));
low_f = J - high_f;

high_o = idct2(high_f);
low_o = idct2(low_f);
figure; imshow(J), colormap(gray); title('mag of DCT');
figure; imshowpair(high_o, low_o, 'montage');

