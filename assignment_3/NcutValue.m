function value = NcutValue(split_point, eig_vector_2, W, D)

x = (eig_vector_2 > split_point);
x = (2 * x) - 1;
d = sum(W,2); 
k = sum(d(x>0))/sum(d);
b = k/(1 - k);
y = (1 + x) - b*(1 - x);

value = (y'*(D - W)*y)/(y'*D*y);

end