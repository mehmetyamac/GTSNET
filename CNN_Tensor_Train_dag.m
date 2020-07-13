function [net, info] = CNN_Tensor_Train_dag(imdb,varargin)

%% Training options (default), change from Main if needed

opts = struct();
% Number of channels
opts.nch = 1;
% Patch size
opts.patchsize = [256 256];
% Measurement rate
opts.MR = 0.1;
% DCT block size
opts.dctblsz = [8 16 32];
% Network directory
opts.expDir = 'train/CNN_Tensor_RDN';
% Batch size
opts.batchSize = 16;
opts.numSubBatches = 1;
% Learning rate
opts.learningRate = [logspace(-3,-3,50) logspace(-4,-4,30) logspace(-5,-5,20)];
% Number of epochs
opts.numEpochs = numel(opts.learningRate);
% Weight decay, momentum
opts.weightDecay = 0.0001;
opts.momentum = 0.5;
% Solver
opts.solver = @solver.adam; % []: SGD solver
opts.gpus = [1]; % []: cpu
opts.plotStatistics = true;

opts = vl_argparse(opts, varargin);

imdb.opts.gpu = ~isempty(opts.gpus);

%% Create the network directory

if ~exist(opts.expDir, 'dir')
    mkdir(opts.expDir);
end

%% Initialize network

Imagesize = [opts.patchsize opts.nch];

% Network design: Compress using block-wise DCTs with different block sizes
net = CNN_Tensor_init('nch',opts.nch,'Imagesize', Imagesize, 'dctblsz', opts.dctblsz, 'MR', opts.MR);

%% Patch and channel size information to imdb

imdb.opts.nch = opts.nch;
imdb.opts.patchsize = opts.patchsize;

%% Train

opts.derOutputs = net.meta.derOutputs;
trainopts = rmfield(opts,{'nch','MR','patchsize','dctblsz'}); % remove options related to the network structure

[net, info] = cnn_train_dag(net, imdb, @getBatch, trainopts);
