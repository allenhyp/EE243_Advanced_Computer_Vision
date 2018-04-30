clc;
close all;
img = imread('./house.tif');
img = double(img(:,:,1));
M = size(img);

[a,h,v,d] = haart2(img, 1);
figure;
subplot(2,2,1), imagesc(a), colormap(gray), title('approximation');
subplot(2,2,2), imagesc(h), title('horizontal detail');
subplot(2,2,3), imagesc(v), title('verticle detail');
subplot(2,2,4), imagesc(d), title('diagonal detail');

[a2,h2,v2,d2] = haart2(a, 1);
figure;
subplot(2,2,1), imagesc(a2), colormap(gray), title('approximation');
subplot(2,2,2), imagesc(h2), title('horizontal detail');
subplot(2,2,3), imagesc(v2), title('verticle detail');
subplot(2,2,4), imagesc(d2), title('diagonal detail');

% drop high frequency (detail) components
d_d2 = drop_high_f_compo(d2);
d_v2 = drop_high_f_compo(v2);
d_h2 = drop_high_f_compo(h2);
d_a2 = drop_high_f_compo(a2);

re_img_1 = idwt2(d_a2,d_h2,d_v2,d_d2,'haar',M);
max(max(abs(re_img_1-a)))

figure;
subplot(1,2,1), imagesc(a), colormap(gray), title('approximation level 1');
subplot(1,2,2), imagesc(re_img_1), colormap(gray), title('reconstruct level 2');

d_d = drop_high_f_compo(d);
d_v = drop_high_f_compo(v);
d_h = drop_high_f_compo(h);
d_a = drop_high_f_compo(a);

re_img_0 = idwt2(d_a,d_h,d_v,d_d,'haar',M);
max(max(abs(re_img_0-img)))

figure;
subplot(1,2,1), imagesc(img), colormap(gray), title('original picture');
subplot(1,2,2), imagesc(re_img_0), colormap(gray), title('reconstruct level 1');



function amplitude_image = drop_high_f_compo(input_image)
    % Compute the 2D fft
    figure;
    f_image = fftshift(fft2(input_image));
    a_image = log(abs(f_image));
    subplot(2,3,1), imshow(a_image, []);
    amplitude_threshold = 9.8;
    bright_spikes = a_image > amplitude_threshold; % Binary image.
    subplot(2,3,2), imagesc(bright_spikes);
    % Exclude the central DC spike.  Everything from row 115 to 143.
    bright_spikes(50:200, :) = 0;
    subplot(2,3,3), imagesc(bright_spikes);
    % Filter/mask the spectrum.
    f_image(bright_spikes) = 0;
    % Take log magnitude so we can see it better in the display.
    a_image_2 = log(abs(f_image));
    min_value = min(min(a_image_2));
    max_value = max(max(a_image_2));
    subplot(2,3,4), imshow(a_image_2, [min_value max_value]);
    filtered_image = ifft2(fftshift(f_image));
    amplitude_image = abs(filtered_image);
    min_value = min(min(amplitude_image));
    max_value = max(max(amplitude_image));
    subplot(2,3,5), imshow(amplitude_image, [min_value max_value]);
end