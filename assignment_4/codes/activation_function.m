function result = activation_function(h)
input = h.';
[m, n] = size(h);
result = zeros(m, n);
for i=1:m
%     a = exp(h(i,:));
    b = sum(exp(h(i,:)));
    for j=1:n
        result(i,j) = exp(h(i,j))/b;
        if isnan(result(i,j))
            result(i,j)=1;
        end
    end
%     result(i,:) = softmax(input(:,i));
%     result(i,:) = translate_max(result(i,:));
end

end