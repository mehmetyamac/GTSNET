function out = reshape_p(in,p)

[sx,sy,~,~] = size(in);

% pad with symmetric
px = mod(p - mod(sx,p), p);
py = mod(p - mod(sy,p), p);
in = padarray(in,[px,py],'symmetric','post');

aa=1:p:sx;
bb=1:p:sy;

[ii,jj]=ndgrid(aa,bb);
out=arrayfun(@(x,y) in(x:x+p-1,y:y+p-1,:),ii,jj,'un',0);

out = cat(4,out{:});

end