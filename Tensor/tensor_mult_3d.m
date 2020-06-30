function [y,dedx,dedt1,dedt2,dedt3] = tensor_mult_3d(x,t1,t2,t3,dedy)

% check if there exists third tensor
if isempty(t3) || nargin < 4
    t3 = eye(size(x,3));
end

%% Forward

sx = size(x);
% First dimension
x1 = t1*reshape(x,sx(1),[]);
sx1 = [size(t1,1), sx(2:end)];
x1 = reshape(x1,sx1);
% Second dimension
x2 = t2*reshape(permute(x1,[2 1 3]), sx1(2), []);
sx2 = [size(t2,1) sx1(1) sx1(3)];
x2 = reshape(x2,sx2);
x2 = permute(x2,[2 1 3]);
sx2 = size(x2);
% Third dimension
x3 = t3*reshape(permute(x2,[3 1 2]), sx2(3), []);
sx3 = [size(t3,1) sx2(1) sx2(2)];
x3 = reshape(x3,sx3);
x3 = permute(x3,[2 3 1]);
sx3 = size(x3);
% Output
y = x3;

%% Backward

if nargin > 4
    % Third dimension
    dedx3 = dedy;
    dedx3 = permute(dedx3,[3 1 2]);
    sx3 = size(dedx3);
    dedx3 = reshape(dedx3,sx3(1),[]);
    dedt3 = dedx3 * reshape(permute(x2,[3 1 2]), sx2(3), [])';
    dedx2 = t3' * dedx3;
    dedx2 = permute(reshape(dedx2, [sx2(3) sx2(1) sx2(2)]),[2 3 1]);
    % Second dimension
    dedx2 = permute(dedx2, [2 1 3]);
    sx2 = size(dedx2);
    dedx2 = reshape(dedx2, sx2(1), []);
    dedt2 = dedx2 * reshape(permute(x1,[2 1 3]), sx1(2), [])';
    dedx1 = t2' * dedx2;
    dedx1 = permute(reshape(dedx1, [sx1(2) sx1(1) sx1(3)]),[2 1 3]);
    % First dimension
    dedx1 = reshape(dedx1,sx1(1),[]);
    dedt1 = dedx1 *  reshape(x,sx(1),[])';   
    dedx = t1' * dedx1;
    dedx = reshape(dedx,sx);
end

end