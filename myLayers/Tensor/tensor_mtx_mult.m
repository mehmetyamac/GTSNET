function [y,dedt] = tensor_mtx_mult(x,t,dim,dedy)

% Number of dimensions
n = max(3,ndims(x));
% Permutation of the dimensions
per = circshift(1:n, -(dim-1));
% Permutation back
per_back = circshift(1:n, (dim-1));

% Permute tensor to bring the dim to front
x = permute(x,per);
% Reshape the tensor
sx = size(x);
x = reshape(x, sx(1), []);

if nargin < 4
    % Forward
    
    % Multiply
    y = t * x;
    % Reshape back
    sy = [size(t,1), sx(2:end)];
    y = reshape(y,sy);
    % Permute back
    y = permute(y,per_back);
    
else
    % Backward
    
    % Permute tensor to bring the dim to front
    dedy = permute(dedy,per);
    % Reshape the tensor
    sy = size(dedy);
    dedy = reshape(dedy,sy(1),[]);
    % Multiply
    y = t' * dedy; % y = dedx
    dedt = dedy * x';
    % Reshape dedx back
    y = reshape(y,sx);
    % Permute dedx back
    y = permute(y,per_back);
    
end

end