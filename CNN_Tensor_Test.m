%%

% Change Matconvnet to your directory
matconvnet = '..\..\..\Coded Aperture\matconvnet-1.0-beta25\matlab';

% Remove layers
rmpath('myLayers');
rmpath('Tensor');
rmpath(genpath('data/utilities'))
run(fullfile(matconvnet, 'vl_setupnn.m'));
% Add layers
addpath('myLayers');
addpath('Tensor');
addpath(genpath('data/utilities'))

%% Options

% Measurement rate
SamplingRatio = 0.01; % Depends on the trained network
% Patch size
patchSize = 128;
% Compute on gpu
useGPU = 0;
% Show results on figure
showResult  = 1;
% Pause time (ms)
pauseTime = 5;

%% Load network

% Network directory
netdir = 'data/trained';
% Epoch
epoch = 'net-epoch-24.mat';
% Load
load(fullfile(netdir,epoch),'net');
% Remove the property 'normalize' from the loss layer. Tihs is due to
% difference in matconvnet versions
l = find(strcmp({net.layers.type},'dagnn.Loss'));
for ll = l
    if isfield(net.layers(ll).block,'normalise')
        net.layers(ll).block = rmfield(net.layers(ll).block,'normalise');
    end
end

net = dagnn.DagNN.loadobj(net);
net.conserveMemory = false;

if useGPU
    net.move('gpu');
end

%% Test dataset

datasets = {'../../Set5'};

%% Test

for dd=1:numel(datasets)
    
    ds = datasets{dd};
    % Image names
    ext         =  {'*.jpg','*.png','*.bmp'};
    filepaths   =  [];
    for i = 1 : length(ext)
        filepaths = cat(1,filepaths, dir(fullfile(ds,ext{i})));
    end
    
    % Initialize PSNR,SSIM values
    PSNRs = zeros(1,length(filepaths));
    SSIMs = zeros(1,length(filepaths));
    
    for i = 1:length(filepaths)
        
        % Read image
        imname = filepaths(i).name;
        image = imread(fullfile(ds,imname));
        image =im2single(image);
        % Ground truth
        label = image;
        
        % Reshape image to process patchwise
        sz = size(image);
        input = reshape_p(image,patchSize);
        
        % Convert to gpu if necessary
        if useGPU
            input = gpuArray(input);
        end
        
        % Process to network
        net.eval({'input',input});
        % Network result
        output=net.getVar('prediction').value;
        
        % Convert back to cpu
        if useGPU
            output = gather(output);
        end
        
        % Reshape back to original image size
        output = reshape_back_p(output,sz,patchSize);
        
        % Calculate psnr,ssim
        PSNR_im = psnr(output,label);
        SSIM_im = ssim(output,label);
        
        % Show results
        if showResult
            figure(1);
            set(gcf,'WindowStyle','docked');
            warning('off','images:imshow:magnificationMustBeFitForDockedFigure');
            subplot(121); imshow(label); title('Ground truth');
            subplot(122); imshow(output); title('Network output');
            
            suptitle(sprintf('%s : PSNR= %2.2f dB, SSIM= %.3f \n', imname, PSNR_im, SSIM_im));
            drawnow;
            pause(pauseTime)
        end
        
        fprintf('%s : PSNR= %2.2f dB, SSIM= %.3f \n', imname, PSNR_im, SSIM_im);
        
        % Save psnr,ssim
        PSNRs(i) = PSNR_im;
        SSIMs(i) = SSIM_im;
    end
    
    % Mean psnr,ssim
    dsname = strsplit(ds,'/'); 
    dsname = dsname{end};
    fprintf('%s mean : PSNR= %2.2f dB, SSIM= %.3f \n', dsname, mean(PSNRs),mean(SSIMs));
    
end