function Y = vl_nnloss(X,c,dzdy,varargin)

opts.loss = 'l2';
opts.regularizer = 'none';
opts.p = 0.8;
opts.alpha = 0.001;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

% --------------------------------------------------------------------
if nargin <= 2 || isempty(dzdy)
    % Forward
    if (strcmp(opts.loss, 'l2'))
        t = ((X-c).^2)/2;
        Y = sum(t(:))/size(X,4); % reconstruction error per sample;
    elseif (strcmp(opts.loss, 'l1'))
        Y = sum(abs(X(:)-c(:))); % L1
    end
    % Regularizer
    if strcmp(opts.regularizer, 'dct')
        r = dctsparsity(X); % regularization
        Y = Y+0.00001*r;
    elseif strcmp(opts.regularizer, 'gradient')
        for ch = 1:size(X,3)
            for n = 1:size(X,4)
                r(:,:,ch,n) = sparse_gradient(X(:,:,ch,n), [], opts.p, opts.alpha, c(:,:,ch,n));
            end
        end
        Y = Y+sum(r(:));
    else
        % No regularizer
    end
    
else
    % Backward
    if (strcmp(opts.loss, 'l2'))
        Y = bsxfun(@minus,X,c).*dzdy;
    elseif (strcmp(opts.loss, 'l1'))
        Y = bsxfun(@times, sign(bsxfun(@minus,X,c)), dzdy);
    end
    % Regularizer
    if strcmp(opts.regularizer, 'dct')
        r = dctsparsity(X,dzdy); % regularization
        Y = Y+0.00001*r;
    elseif strcmp(opts.regularizer, 'gradient')
        for ch = 1:size(X,3)
            for n = 1:size(X,4)
                r(:,:,ch,n) = sparse_gradient(X(:,:,ch,n), dzdy, opts.p, opts.alpha, c(:,:,ch,n));
            end
        end
        Y = Y+r;
    else
        % No regularizer
    end
end

end


function y = sparse_gradient(x, dzdy, p, alpha, l)

backMode = ~isempty(dzdy);

[Gx_x,Gy_x] = imgradientxy(x,'intermediate');
[Gx_l,Gy_l] = imgradientxy(l,'intermediate');
Dx_l = exp(-10*(abs(Gx_l).^p));
Dy_l = exp(-10*(abs(Gy_l).^p));

if backMode
    % Derivative
    wx = Dx_l(:,1:end-1);
    wxplus = padarray(wx,[0 1],0,'pre');
    wxminus = padarray(wx,[0 1],0,'post');
    
    wy = Dy_l(1:end-1,:);
    wyplus = padarray(wy,[1 0],0,'pre');
    wyminus = padarray(wy,[1 0],0,'post');
    
    Gpx = (abs(Gx_x)+eps).^(p-1);
    Gpx = Gpx(:,1:end-1);
    Gpxplus = padarray(Gpx,[0 1],0,'pre');
    Gpxminus = padarray(Gpx,[0 1],0,'post');
    
    Gsx = sign(Gx_x);
    Gsx = Gsx(:,1:end-1);
    Gsxplus = padarray(Gsx,[0 1],0,'pre');
    Gsxminus = padarray(Gsx,[0 1],0,'post');
    
    Gpy = (abs(Gy_x)+eps).^(p-1);
    Gpy = Gpy(1:end-1,:);
    Gpyplus = padarray(Gpy,[1 0],0,'pre');
    Gpyminus = padarray(Gpy,[1 0],0,'post');
    
    Gsy = sign(Gy_x);
    Gsy = Gsy(1:end-1,:);
    Gsyplus = padarray(Gsy,[1 0],0,'pre');
    Gsyminus = padarray(Gsy,[1 0],0,'post');
    
    y = dzdy * alpha * p * (wxplus.*Gpxplus.*Gsxplus + wyplus.*Gpyplus.*Gsyplus...
        - wxminus.*Gpxminus.*Gsxminus - wyminus.*Gpyminus.*Gsyminus);
else
    Dx_x = abs(Gx_x).^p;
    Dy_x = abs(Gy_x).^p;
    y = alpha * (Dx_l(:)'*Dx_x(:) + Dy_l(:)'*Dy_x(:));
end
end




function y = dctsparsity(x,dzdy)
y = zeros(size(x),'single');
if nargin <= 2 || isempty(dzdy)
    for ch = 1:size(x,3)
        for n = 1:size(x,4)
            y(:,:,ch,n) = dct2(x(:,:,ch,n));
        end
    end
    y = sum(abs(y(:)));
else
    for ch = 1:size(x,3)
        for n = 1:size(x,4)
            y(:,:,ch,n) = idct2(sign(dct2(x(:,:,ch,n))));
        end
    end
    y = bsxfun(@times, y, dzdy);
end
end