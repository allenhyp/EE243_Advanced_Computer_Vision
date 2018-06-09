function points_3d = get3DPoints(mp1, mp2, cm1, cm2)
% function points_3d = get3DPoints(points_2d)
    % [m, n] = size(points_2d);
    % points_3d = zeros(n, 3);
    % mean subtraction
    % w = points_2d * (eye(n, n)-ones(n, n)/n);
    % [~, S, V] = svd(w);
    % points_3d = sqrt(S(1:3, 1:3))*V(:, 1:3)';
    [~, n] = size(mp1);
%     points_3d = triangulate(mp1, mp2, cm1, cm2);
    points_3d = zeros(n, 3);
    for i = 1:n
        points_3d(i, :) = triangulate_one(mp1(:, i), mp2(:, i), cm1, cm2);
    end
end

function points_3d_i = triangulate_one(p1, p2, cm1, cm2)
    P = zeros(4, 4);
    P(1:2, :) = p1*cm1(3, :) - cm1(1:2, :);
    P(3:4, :) = p2*cm2(3, :) - cm2(1:2, :);
    [~, ~, V] = svd(P);
    v = V(:, length(V));
    v = v/v(length(v));
    points_3d_i = v(1:3)';
end
