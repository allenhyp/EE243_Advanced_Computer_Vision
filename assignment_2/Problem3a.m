close all;
img_house = imread('./house.tif');
img_house = uint8(img_house(:,:,1));
img_lena = imread('./lena_gray_256.tif');
col = 5;
threshold = 0.003;
count = 1;
figure;

for delta=0.0:0.001:0.004
    bw_house_log = edge(img_house,'log',threshold+delta);
    bw_house_canny = edge(img_house,'Canny',[(threshold+delta)*10 (threshold+delta)*50]);

    subplot(col,2,count), imshow(bw_house_log);
    subplot(col,2,count+1), imshow(bw_house_canny);
    count = count+2;
end

count = 1;
figure;
for delta=0.0:0.001:0.004
    bw_lena_log = edge(img_lena,'log',threshold+delta);
    bw_lena_canny = edge(img_lena,'Canny',[(threshold+delta)*10 (threshold+delta)*100]);

    subplot(col,2,count), imshow(bw_lena_log);
    subplot(col,2,count+1), imshow(bw_lena_canny);
    count = count+2;
end