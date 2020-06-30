function inputs = getBatch(imdb,batch)

images = vl_imreadjpeg(imdb.inputs(batch),'Resize',imdb.opts.patchsize);
images = cat(4,images{:});
images = images./255;
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