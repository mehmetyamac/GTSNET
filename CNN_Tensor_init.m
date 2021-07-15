function net = CNN_Tensor_init( varargin )
%CNN_Tensor_init: Initialize network, including tensor-based sensing and
%reconstruction, together with the RDNN-based post-processing

%% Initialize Network

net = dagnn.DagNN();

%% Default network options

opts = struct();
% Compression (Measurement) ratio
opts.MR = 0.1;
% Number of image channels (1: grayscale, 3: RGB)
opts.nch = 3;
% Default image Size
opts.Imagesize=[256,256,opts.nch];
% Block sizes to be used in block-wise DCT
opts.dctblsz = [8 16 32];

opts = vl_argparse(opts, varargin);

if isempty(opts.dctblsz); opts.dctblsz = [0 0 0]; end

% Derivatives
net.meta.derOutputs = {'cost_func',1};

%% Sampling

Imagesize = opts.Imagesize;
dctblsz = opts.dctblsz;


% Measurement ratio in each dimension
mr1 = sqrt(opts.MR);
MR=[mr1, mr1, 1];

% Tensor sizes
Ysize=ceil(MR.*Imagesize); % Measurement
Asize=[Ysize' Imagesize']; % Proxy

nscale = numel(dctblsz);
% Size of DCT matrices
Dsize = [Imagesize' Imagesize'];

CSoutput = cell(1,nscale);
Dblocks = cell(1,nscale);
Ablocks = cell(1,nscale);

for i = 1:nscale
    
    if dctblsz(i) == 0 % No DCT
        % Tensor layer
        A = initialize_tensor(Asize);
        
        if opts.nch == 1 %grayscale
            A{3} = single(1);
        end
        
        block = Tensor_3d('tensorsize',Asize, 'inittensor', A);
        name = ['CS_' num2str(i)];
        paramname = {['A1_' num2str(i)], ['A2_' num2str(i)], ['A3_' num2str(i)]};
        net.addLayer(name,block,'input',['Y_' num2str(i)],paramname);
        
        if opts.nch == 1 %grayscale
            net.params(net.getParamIndex(['A3_' num2str(i)])).learningRate = 0;
        end
        
        CSoutput{i} = ['Y_' num2str(i)];
        % Save it to use later
        Ablocks{i} = A;
    else % DCT
        bsize = [dctblsz(i) dctblsz(i) opts.nch];
        % DCT with changing block
        D = dct_init(Imagesize,bsize);
        block = Tensor_3d('tensorsize',Dsize, 'inittensor', D);
        name = ['DCT_' num2str(dctblsz(i))];
        outname = ['DCT_' num2str(dctblsz(i))];
        paramname = {['D1_' num2str(dctblsz(i))], ['D2_' num2str(dctblsz(i))], ['D3_' num2str(dctblsz(i))]};
        net.addLayer(name,block,'input',outname,paramname);
        for p = 1:numel(paramname)
            net.params(net.getParamIndex(paramname{p})).learningRate = 0;
        end
        Dblocks{i} = D;
        
        % Tensor layer
        A = initialize_tensor(Asize);
        
        if opts.nch == 1 %grayscale
            A{3} = single(1);
        end
        
        block = Tensor_3d('tensorsize',Asize, 'inittensor', A);
        name = ['CS_' num2str(dctblsz(i))];
        paramname = {['A1_' num2str(dctblsz(i))], ['A2_' num2str(dctblsz(i))], ['A3_' num2str(dctblsz(i))]};
        net.addLayer(name,block,outname,['Y_' num2str(dctblsz(i))],paramname);
        
        if opts.nch == 1 %grayscale
            net.params(net.getParamIndex(['A3_' num2str(dctblsz(i))])).learningRate = 0;
        end
        
        CSoutput{i} = ['Y_' num2str(dctblsz(i))];
        % Save it to use later
        Ablocks{i} = A;
    end
    
end

% Sum
net.addLayer('CS',dagnn.Sum(),CSoutput,'CS');


%% Initial reconstruction

adjAsize=fliplr(Asize);
IDsize=fliplr(Dsize);

for i = 1:nscale
    
    if dctblsz(i) == 0 % No DCT
        %adjoint layer
        adjA = cellfun(@(T) T', Ablocks{i}, 'UniformOutput', false);
        block = Tensor_3d('tensorsize',adjAsize, 'inittensor', adjA);
        name = ['adjCS_' num2str(i)];
        outname = ['proximal_' num2str(i)];
        paramname = {['adjA1_' num2str(i)], ['adjA2_' num2str(i)], ['adjA3_' num2str(i)]};
        net.addLayer(name,block,'CS',outname,paramname);
        
        if opts.nch == 1 %grayscale
            net.params(net.getParamIndex(['adjA3_' num2str(i)])).learningRate = 0;
        end
        
        adjCSoutput{i} = outname;
    else % DCT
        %adjoint layer
        adjA = cellfun(@(T) T', Ablocks{i}, 'UniformOutput', false);
        block = Tensor_3d('tensorsize',adjAsize, 'inittensor', adjA);
        name = ['adjCS_' num2str(dctblsz(i))];
        outname = ['proximalDCT_' num2str(dctblsz(i))];
        paramname = {['adjA1_' num2str(dctblsz(i))], ['adjA2_' num2str(dctblsz(i))], ['adjA3_' num2str(dctblsz(i))]};
        net.addLayer(name,block,'CS',outname,paramname);
        
        if opts.nch == 1 %grayscale
            net.params(net.getParamIndex(['adjA3_' num2str(dctblsz(i))])).learningRate = 0;
        end
        
        %IDCT
        ID = cellfun(@(T) T', Dblocks{i}, 'UniformOutput', false);
        block = Tensor_3d('tensorsize',IDsize, 'inittensor', ID);
        name = ['IDCT_' num2str(dctblsz(i))];
        paramname = {['ID1_' num2str(dctblsz(i))], ['ID2_' num2str(dctblsz(i))], ['ID3_' num2str(dctblsz(i))]};
        net.addLayer(name,block,outname,['IDCT_' num2str(dctblsz(i))],paramname);
        for p = 1:numel(paramname)
            net.params(net.getParamIndex(paramname{p})).learningRate = 0;
        end
        adjCSoutput{i} = ['IDCT_' num2str(dctblsz(i))];
    end
end

% Sum
net.addLayer('adjCS',dagnn.Sum(),adjCSoutput,'proximal');
output = 'proximal';

%% Deep reconstruction

net = RDN_network(net,opts.nch,output,'prediction');

%% Loss

% Proximal loss
net.addLayer('loss_proximal',dagnn.Loss('loss','l1','opts',{}),{'proximal','ground'},'cost_func_proximal')

% Prediction loss
opts_loss = {'regularizer','gradient','alpha',0.005,'p',0.9};
net.addLayer('loss_pred',dagnn.Loss('loss','l1','opts',opts_loss),{'prediction','ground'},'cost_func_pred')

% Global loss
net.addLayer('global_loss',dagnn.Sum(),{'cost_func_pred','cost_func_proximal'},'cost_func');

%% Initialize parameters

net.initParams();
