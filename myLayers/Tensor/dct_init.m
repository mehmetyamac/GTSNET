function T = dct_init(imsz,blsz)

% image size to block size ratio should be integer

nch = numel(imsz);

T = cell(1,nch);
%%create measurements
for c = 1:nch
    D = dctmtx(blsz(c));
    Tc = D;
    for i = 1:(imsz(c)/blsz(c))-1
        Tc = blkdiag(Tc,D);
    end
    T{c} = single(Tc);
end


end