
% Change Matconvnet to your directory
matconvnet = '..\matconvnet\matlab';

% You can compile Matconvnet (one time only) with gpu using the following code:
% wd = pwd;
% cd(matconvnet);
% vl_compilenn('EnableGpu',true,'EnableImreadJpeg',true);
% cd(wd);

% Remove layers
rmpath(genpath('myLayers'));
run(fullfile(matconvnet, 'vl_setupnn.m'));
% Add layers
addpath(genpath('myLayers'));

%% Create imdb

imdb=struct;
imdb.opts = [];
% Folder to train and validation data
TrainData = 'data/TrainingData/';
ValData = 'data/ValidateData/';

%% Important parameters

patchsize = [256 256]; % Patch size
nch = 1; % Number of channels, (1: gray, 3: rgb)
ctype = 'gray'; % 'gray' or 'rgb'
MR = 0.2; % Measurement rate
gtsnet = 'GTSNET-3';
dctblsz = [8 16 32]; % DCT block size: [0 0 0], or [] means no DCT

batchSize = 16; % Batch size

% Network directory
MRname = ['MR_' strrep(num2str(MR),'.','')];
netdir = fullfile('trained',ctype,gtsnet,MRname);

learningRate = [logspace(-3,-3,50) logspace(-4,-4,30) logspace(-5,-5,20)]; % Learning Rate

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
opts.plotStatistics = false;

[net, info] = CNN_Tensor_Train_dag(imdb,opts);
