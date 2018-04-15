clc;
clear all;
close all;

I = imread('./Assigment-1/TEST_IMAGES/house.tif');
img = I(:,:,1);
% figure; imshow(img); title('original_img');
img = single(imresize(img, 0.3));
for k = 0.1:0.5:1.1
    img = imgaussfilt(img,k);
    [m,n] = size(img);
    output = zeros(m,n);
    sum_inner = 0;
    h = waitbar(0, 'Calculating...');

    for l = 0:m-1
        for k = 0:n-1
            for x = 0:m-1
                for y = 0:n-1
                    sum_inner = sum_inner + img(x+1,y+1) * exp(-1i*2*3.1416*(k*x/m+l*y/n));
                end
            end
            output(k+1,l+1) = sum_inner;
            sum_inner = 0;
        end
        waitbar(l/m);
    end

    close(h)
    output2 = output*255;
    figure; imshow(output2); title('dft plot');
    output3 = zeros(m,n);
    for u = 1:m
        for v = 1:n
            output3(u,v) = sqrt((real(output2(u,v))^2 + imag(output(u,v))^2))/1000000;
        end
    end
    figure; imshow(output3); colormap(jet); colorbar; title('abs value of dft plot');
end

