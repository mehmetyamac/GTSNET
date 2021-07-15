function T = dct_init_overlap(imsz,blsz,stride)

nch = numel(imsz);
T = cell(1,nch);

for c = 1:nch
    
    Dsize = ceil((imsz(c)-blsz(c)+1)/stride(c))*blsz(c);    
    D = dctmtx(blsz(c));
    Db = zeros(Dsize,imsz(c));
    
    for i = 1:blsz(c)
        C1 = convmtx(D(i,end:-1:1)',imsz(c));
        C1 = C1(blsz(c):end-blsz(c)+1,:);
        C1 = C1(1:stride(c):end,:);
        Db(i:blsz(c):end,:) = C1;
    end
    T{c} = Db;
end

end