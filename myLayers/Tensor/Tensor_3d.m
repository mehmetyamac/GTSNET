classdef Tensor_3d < dagnn.Layer
    % Tensor multiplaction layer
    % y = T * X, where
    % T = tensor(t1,t2,t3)
    
    properties
        opts = struct();
        tensorsize = [50 100;50 100;20 50];
        nTdims = 3; %number of tensor dimensins
        inittensor = [];
    end
    
    methods
        
        function outputs = forward(obj, inputs, params)
            
            % input
            x = inputs{1};
            % Tensors
            t1 = params{1};
            t2 = params{2};
            t3 = params{3};
            
            isgpu = isa(x, 'gpuArray');
            if (isgpu)
                x = gpuArray(single(x));
                t1 = gpuArray(single(t1));
                t2 = gpuArray(single(t2));
                t3 = gpuArray(single(t3));
            else
                x = single(x);
                t1 = single(t1);
                t2 = single(t2);
                t3 = single(t3);
            end
                        
            % Main
            y1 = tensor_mtx_mult(x,t1,1);
            y2 = tensor_mtx_mult(y1,t2,2);
            y = tensor_mtx_mult(y2,t3,3);
            
            % output
            outputs{1} = y;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            
            % input
            x = inputs{1};
            % Tensors
            t1 = params{1};
            t2 = params{2};
            t3 = params{3};
            % Output derivative
            dedy = derOutputs{1};
            
            isgpu = isa(x, 'gpuArray');
            if (isgpu)
                x = gpuArray(single(x));
                t1 = gpuArray(single(t1));
                t2 = gpuArray(single(t2));
                t3 = gpuArray(single(t3));
                dedy = gpuArray(single(dedy));
            else
                x = single(x);
                t1 = single(t1);
                t2 = single(t2);
                t3 = single(t3);
                dedy = single(dedy);
            end
            
            % number of batches
            N = size(x, 4);
                        
            % Main
            % Forward
            y21 = tensor_mtx_mult(x,t1,1);
            y22 = tensor_mtx_mult(y21,t2,2);
            %y = tensor_mtx_mult(y22,t3,3);
            % Backward
            [dedx3,dedt3] = tensor_mtx_mult(y22,t3,3,dedy);
            [dedx2,dedt2] = tensor_mtx_mult(y21,t2,2,dedx3);
            [dedx,dedt1] = tensor_mtx_mult(x,t1,1,dedx2);
            
            % Average over batches
            dedt1 = dedt1/N;
            dedt2 = dedt2/N;
            dedt3 = dedt3/N;
            
            % Outputs
            derInputs{1} = dedx;
            derParams{1} = dedt1;
            derParams{2} = dedt2;
            derParams{3} = dedt3;
            
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            sz = obj.tensorsize;
            outputSizes{1} = [sz(:,1)' inputSizes{1}(4)];
        end
        
        function params = initParams(obj)
            % If initial parameters are defined, use them
            if ~isempty(obj.inittensor)
                params = obj.inittensor;
            else
                % Assign Gaussian random
                tsize = obj.tensorsize;
                params{1} = randn(tsize(1,:),'single'); % t1
                params{2} = randn(tsize(2,:),'single'); % t2
                params{3} = randn(tsize(3,:),'single'); % t3
            end
        end
        
        function obj = Tensor_3d(varargin)
            obj.load(varargin);
        end
    end
end