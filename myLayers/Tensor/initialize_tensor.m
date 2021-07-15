function T = initialize_tensor(tsize)
% Random measiurements for now

T = cell(1,size(tsize,1));
%%create measurements
for c = 1:size(tsize,1)
    m = tsize(c,1);
    n = tsize(c,2);
    %tmp = dct(eye(n));
    rndn = single(sqrt( 1 /m)*randn(m, n));
    %rndn(1:round(m/2),:) = tmp(1:round(m/2),:);
    T{c} = rndn;
end
    
end