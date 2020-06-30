function [net, info] = CNN_Tensor_Train_dag(imdb,varargin)

%% Training options

opts = struct();
opts.expDir = 'train/CNN_Tensor_RDN';
opts.batchSize = 16;
opts.numSubBatches = 1;
opts.learningRate = [logspace(-3,-3,50) logspace(-4,-4,30) logspace(-5,-5,20)];
opts.numEpochs = numel(opts.learningRate);
opts.weightDecay = 0.0001;
opts.momentum = 0.5;
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

dctblsz = [8 16 32];
Imagesize = [256 256 3];

% Network design: Compress using block-wise DCTs with different block sizes
net = CNN_Tensor_init('Imagesize', Imagesize, 'dctblsz', dctblsz);

%% Train

opts.derOutputs = net.meta.derOutputs;
[net, info] = cnn_train_dag(net, imdb, @getBatch, opts);
