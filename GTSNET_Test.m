%%

% Change Matconvnet to your directory
matconvnet = '..\matconvnet\matlab';

% Remove layers
rmpath(genpath('myLayers'));
rmpath(genpath('data/utilities'))
run(fullfile(matconvnet, 'vl_setupnn.m'));
% Add layers
addpath(genpath('myLayers'));
addpath(genpath('data/utilities'))

%% Options

% Measurement rate
MRs = [0.01 0.05 0.1];
gtsnet = 'GTSNET-3';
saveimage=0;
% Patch size
patchSize = 256;
% Compute on gpu
useGPU = 0;
% Show results on figure
showResult  = 1;
% Pause time (ms)
pauseTime = 0;

nch = 1;
ctype='gray'; % 'rgb' or 'gray'

%% Load network

for ss = 1:numel(MRs)
    
    % Network directory
    MRname = ['MR_' strrep(num2str(MRs(ss)),'.','')];
    netdir = fullfile('trained',ctype,gtsnet,MRname);
    
    % Epoch
    epoch = 'net-epoch-100.mat';
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
    
    datasets = {'Set5','Set11'};
    
    %% Test
    
    for dd=1:numel(datasets)
        
        resfolder = fullfile('results', ctype, gtsnet, MRname, datasets{dd});
        if ~exist(resfolder, 'dir')
            mkdir(resfolder);
        end
        
        ds = fullfile('data/TestData/',datasets{dd});
        % Image names
        ext         =  {'*.jpg','*.png','*.bmp','*.tif'};
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
            image = im2single(image);
            if nch ==1
                if size(image,3)==3
                    image = rgb2ycbcr(image);
                    image = image(:, :, 1);
                end
            end
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
                
                title(sprintf('%s : PSNR= %2.2f dB, SSIM= %.3f \n', imname, PSNR_im, SSIM_im));
                
                drawnow;
                pause(pauseTime)
            end
            if saveimage==1
                pth1= fullfile(resfolder,[imname,'_',num2str(PSNR_im,'%2.2f'),'dB','_',num2str(SSIM_im,'%2.4f'), '.png']);
                imwrite(im2uint8(output),pth1);
            end
            
            fprintf('%s : PSNR= %2.2f dB, SSIM= %.3f \n', imname, PSNR_im, SSIM_im);
            
            % Save psnr,ssim
            PSNRs(i) = PSNR_im;
            SSIMs(i) = SSIM_im;
        end
        
        % Mean psnr,ssim
        dsname = strsplit(ds,'/');
        dsname = dsname{end};
        mean_psnr = mean(PSNRs);
        mean_ssim = mean(SSIMs);
        fprintf('MR = %.f %% %s mean : PSNR= %2.2f dB, SSIM= %.3f \n', MRs(ss)*100, dsname, mean_psnr, mean_ssim);
        
        dname = {filepaths.name}';
        resfile = fullfile(resfolder,'results.mat');
        save(resfile,'dname','PSNRs','SSIMs','mean_psnr','mean_ssim');
        
    end
    
end
