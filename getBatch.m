function inputs = getBatch(imdb,batch)

images = vl_imreadjpeg(imdb.images.inputs(batch),'Resize',imdb.opts.patchsize);
images = cat(4,images{:});
images = images./255;
% Gray scale
if imdb.opts.nch == 1
    % convert to ycbcr
    for i = 1:size(images,4)
        images(:,:,:,i) = rgb2ycbcr(images(:,:,:,i));
    end
    % Take y channel only
    images = images(:,:,1,:);
end
labels = images;
if opts.numGpus > 0
    images = gpuArray(images) ;
    labels = gpuArray(labels) ;
end
GTDCT_scales = {};
if isfield(imdb.opts, 'dctblsz')
    dctblsz = imdb.opts.dctblsz;
    nscale = numel(dctblsz);
    Dblocks = imdb.opts.Dblocks;
    for i = 1:nscale
        Di = Dblocks{i};
        gtdct = images;
        for c = 1:numel(Di)
            gtdct = tensor_mtx_mult(gtdct,Di{c},c);
        end
        GTDCT_scales = [GTDCT_scales {['GTDCT_' num2str(dctblsz(i))], gtdct}];
    end
end
inputs = [{'input', images, 'ground', labels} GTDCT_scales];


end