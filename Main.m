%%

% Change Matconvnet to your directory
matconvnet = '..\..\Coded Aperture\matconvnet-master\matlab';

% Remove layers
rmpath('myLayers');
rmpath('Tensor');
run(fullfile(matconvnet, 'vl_setupnn.m'));
% Add layers
addpath('myLayers');
addpath('Tensor');

%% Create imdb

imdb=struct;
imdb.opts = [];
TrainData = './data/TrainingData/';
ValData = './data/ValidateData/';

fext = '*.png';
dtrain = dir([TrainData fext]);
dval = dir([ValData fext]);
train = {dtrain.name};
train = cellfun(@(d) fullfile(TrainData,d), train, 'UniformOutput', false);
val = {dval.name};
val = cellfun(@(d) fullfile(ValData,d), val, 'UniformOutput', false);

set_train = ones(1,numel(dtrain),'single');
set_val = 2*ones(1,numel(dval),'single');

imdb.inputs = [train val];
imdb.set = [set_train set_val];

%% Train

[net, info] = cnn_Tensor_Train_dag(imdb);