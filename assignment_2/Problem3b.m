close all;
img_house = imread('./house.tif');
img_house = uint8(img_house(:,:,1));
bw_img = edge(img_house,'Canny',[0.25 0.30]);

teta_step = 1;
d_step = 1;
thresh = 3;
d = 1:d_step:sqrt((size(bw_img,1))^2+(size(bw_img,2))^2);
teta = 0:teta_step:180-teta_step;

accu = zeros(length(d),length(teta));
[y x] = find(bw_img);
for count = 1:numel(x)
    iter_teta = 0;
    for tetai = teta*pi/180
        iter_teta = iter_teta+1;
        roi = x(count)*cos(tetai)+y(count)*sin(tetai);
        if roi >= 1 && roi <= d(end)
            temp = abs(roi-d);
            mintemp = min(temp);
            iter_d = find(temp == mintemp);
            iter_d = iter_d(1);
            accu(iter_d,iter_teta) = accu(iter_d,iter_teta)+1;
        end
    end
end
figure; imagesc(accu);
accu_max = imregionalmax(accu);
[potential_d potential_teta] = find(accu_max == 1);
accu_new = accu - thresh;
peak = [];
teta_detect = [];
for count = 1:numel(potential_d)
    if accu_new(potential_d(count),potential_teta(count)) >= 0
        peak = [peak;potential_d(count)];
        teta_detect = [teta_detect;potential_teta(count)];
    end
end

peak = peak * d_step;
teta_detect = teta_detect *teta_step - teta_step;

