img = imread('./house.tif');
img = double(img(:,:,1));
M = size(img);

[c,s]=wavedec2(img,2,'haar');
[H1,V1,D1] = detcoef2('all',c,s,1);
A1 = appcoef2(c,s,'haar',1); 
V1img = wcodemat(V1,255,'mat',1);
H1img = wcodemat(H1,255,'mat',1);
D1img = wcodemat(D1,255,'mat',1);
A1img = wcodemat(A1,255,'mat',1);
[H2,V2,D2] = detcoef2('all',c,s,2);
A2 = appcoef2(c,s,'haar',2); 
V2img = wcodemat(V2,255,'mat',1);
H2img = wcodemat(H2,255,'mat',1);
D2img = wcodemat(D2,255,'mat',1);
A2img = wcodemat(A2,255,'mat',1);
figure;
subplot(2,2,1);
imagesc(A1img);
colormap(gray);
title('Approximation Coef. of Level 1');
subplot(2,2,2);
imagesc(H1img);
title('Horizontal detail Coef. of Level 1');
subplot(2,2,3);
imagesc(V1img);
title('Vertical detail Coef. of Level 1');
subplot(2,2,4);
imagesc(D1img);
title('Diagonal detail Coef. of Level 1');


figure;
subplot(2,2,1);
imagesc(A2img);
colormap(gray);
title('Approximation Coef. of Level 2');
subplot(2,2,2);
imagesc(H2img);
title('Horizontal detail Coef. of Level 2');
subplot(2,2,3);
imagesc(V2img);
title('Vertical detail Coef. of Level 2');
subplot(2,2,4);
imagesc(D2img);
title('Diagonal detail Coef. of Level 2');



