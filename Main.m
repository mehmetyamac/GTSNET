%%

% Change Matconvnet to your directory
matconvnet = '..\..\..\Coded Aperture\matconvnet-1.0-beta25\matlab';

% Remove layers
rmpath('myLayers');
rmpath('Tensor');
run(fullfile(matconvnet, 'vl_setupnn.m'));
% Add layers
addpath('myLayers');
addpath('Tensor');

%% Important parameters

patchsize = [256 256]; % Patch size
nch = 1; % Number of channels
MR = 0.1; % Measurement rate
dctblsz = [8 16 32]; % DCT block size

batchSize = 32; % Batch size
netdir = 'train/CNN_Tensor_RDN'; % Network directory
learningRate = [logspace(-3,-3,50) logspace(-4,-4,30) logspace(-5,-5,20)]; % Learning Rate

%% Create imdb

imdb=struct;
imdb.opts = [];
TrainData = './data/TrainingData/';
ValData = './data/ValidateData/';

fext = '*.jpeg';
dtrain = dir([TrainData fext]);
dval = dir([ValData fext]);
train = {dtrain.name};
train = cellfun(@(d) fullfile(TrainData,d), train, 'UniformOutput', false);
val = {dval.name};
val = cellfun(@(d) fullfile(ValData,d), val, 'UniformOutput', false);

set_train = ones(1,numel(dtrain),'single');
set_val = 2*ones(1,numel(dval),'single');

imdb.images.inputs = [train val];
imdb.images.set = [set_train set_val];

%% Train

% Options
opts = struct();
opts.nch = nch;
opts.patchsize = patchsize;
opts.MR = MR;
opts.dctblsz = dctblsz;
opts.expDir = netdir;
opts.batchSize = batchSize;
opts.numSubBatches = 1;
opts.learningRate = learningRate;
% Number of epochs
opts.numEpochs = numel(opts.learningRate);
% Weight decay, momentum
opts.weightDecay = 0.0001;
opts.momentum = 0.5;
% Solver
opts.solver = @solver.adam; % []: SGD solver
opts.gpus = [1]; % []: cpu
opts.plotStatistics = true;


[net, info] = CNN_Tensor_Train_dag(imdb,opts);
