function out = reshape_back_p(in,sz,p)

if numel(sz)==1
    sz = [sz sz];
end
sx = sz(1);
sy = sz(2);

% find pad size
px = mod(p - mod(sx,p), p);
py = mod(p - mod(sy,p), p);

n = (sx+px)/p;
m = (sy+py)/p;

aa=1:n;
bb=1:m;

[ii,jj]=ndgrid(aa,bb);
out=arrayfun(@(x,y) in(:,:,:,(y-1)*n+x),ii,jj,'un',0);
out = cell2mat(out);
out = out(1:sx,1:sy,:,:);

end