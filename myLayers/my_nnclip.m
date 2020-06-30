function y = my_nnclip(x,varargin)
% Clip the values less than 0 and more than 1

if ~isempty(varargin) && ~ischar(varargin{1})  % passed in dzdy
  dzdy = varargin{1} ;
  varargin(1) = [] ;
else
  dzdy = [] ;
end

opts.leak = [0 0];
opts.leaktype = 'symmetric';
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

if(numel(opts.leak) == 1)
    switch lower(opts.leaktype)
        case {'symmetric', 'both'}
            opts.leak = [opts.leak opts.leak];
        case 'pre'
            opts.leak = [opts.leak 0];
        case 'post'
            opts.leak = [0 opts.leak];
    end
end

if all(opts.leak == 0)
  if nargin <= 1 || isempty(dzdy)
    y = min(max(x, 0), 1);
  else
    y = dzdy .* (x > 0 & x < 1);
  end
else
  if nargin <= 1 || isempty(dzdy)
    y = x .* (opts.leak(1) + (1 - opts.leak(1)) * (x > 0));
    y = y .* (x < 1) + (y .* opts.leak(2) + 1 - opts.leak(2)) .* (x >= 1);
  else
    y = dzdy .* (opts.leak(1) + (1 - opts.leak(1)) * (x > 0));
    y = y .* (opts.leak(2) + (1 - opts.leak(2)) .* (x < 1) );
  end
end
