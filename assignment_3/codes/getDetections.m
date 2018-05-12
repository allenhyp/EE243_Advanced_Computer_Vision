function bbox = getDetections(D)

% D is a sum of difference image
% bbox is a N x 4 matrix, containing the x,y,w,h of each bbox and N is the number of bbox

% YOUR CODE HERE. DO NOT CHANGE ANYTHING ABOVE THIS.
    d = D./max(D(:,:));
    d = medfilt2(d, [5, 5]) > 0.5;
    figure; imshow(d*255);
    se = strel('disk', 10);
    c = imclose(d, se);
    CC = bwconncomp(c);
    stats = regionprops(CC, 'BoundingBox');
    figure; imshow(c*255); hold on
    for i = 1:length(stats)
        cur_box = stats(i).BoundingBox;
        rectangle('Position', [cur_box(1), cur_box(2), cur_box(3), cur_box(4)], 'EdgeColor','r','LineWidth',2);
        bbox(i,:) = cur_box;
    end
    hold off
end