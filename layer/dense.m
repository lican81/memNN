% Copyright (c) 2018, 
% Department of Electrical and Computer Engineering, University of Massachusetts Amherst
% All rights reserved.
% 
% Created by 
%     Can Li (Email: me at ilican dot com
%             Website: http://ilican.com)
%     Daniel Belkin, Zhongrui Wang, Wenhao Song
% 
% PI: Prof. Qiangfei Xia (Email: qxia at umass dot com
%                         Website: http://nano.ecs.umass.edu)
%     Prof. J. Joshua Yang (Email: jjyang at umass dot com
%                         Website: http://www.ecs.umass.edu/ece/jjyang/)
% 
% LICENSE
% 
% The MIT License (MIT)
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

classdef dense < layer
    % This class provides various activation functions
    properties
        nlayer %current layer number
        backend
        
        bias_config % a two dimensional array: [ratio rep]
        
        input_dim
        output_dim
        weight_dim
        
        act_name
        
        x_in_history
        y_out_history
    end
    
    methods
        function obj = dense( output_dim, varargin )
            % DENSE the construction funciton for a fully connect layer
            %
            
            okargs = {'input_dim', 'activation', 'bias_config'};
            defaults = {NaN, 'linear', [1 1]};
            [obj.input_dim, obj.act_name, obj.bias_config] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            obj.output_dim = output_dim;
            
            obj.set_weight_dim();
            obj.initialize();
        end
        
        function initialize(obj)
            obj.x_in_history = [];
            obj.y_out_history = [];
        end
        
        function set_weight_dim(obj)
            obj.weight_dim = [obj.output_dim obj.input_dim + obj.bias_config(2) ];
        end
        
        function y_out = call(obj, x_in)
            % CALL The forward pass of the layer
            % Input:
            %   x_in:   The input of the layer
            % Output:
            %   y_out:  The output
            %
            
            n = size(x_in, 2); % The batch size
            
            % Add bias to the input.
            %   The bias_config  = [ratio nrep]
            x_in_full = [x_in; repmat( obj.bias_config(1), obj.bias_config(2), n) ];
            
            % Forward propogation
            y_out = obj.backend.multiply( x_in_full, obj.nlayer);
            
            % Activation
            act = activations.get( obj.act_name, 'act' );
            y_out = act( y_out );
            
            % Store the output for future backpropogation 
            obj.y_out_history = cat(3, obj.y_out_history, y_out);
            obj.x_in_history = cat(3, obj.x_in_history, x_in_full);
        end
        
        function [grads, dx] = calc_gradients(obj, dy)
            % CALC_GRADIENTS: The backward pass of the layer
            % Input:
            %   dy:     The delta on this layer
            % Output:
            %   dW:     The weight gradient in this layer
            %   dx:     The delta for previous layer
            %
            
            % Calculate the delta before activation
            [y_out,  obj.y_out_history] = obj.history_pop( obj.y_out_history);
            [x_in,  obj.x_in_history] = obj.history_pop( obj.x_in_history);
            
            % Activation
            if ~contains( obj.act_name, 'softmax') 
                act = activations.get( obj.act_name, 'deriv' );
                dy = dy .* act( y_out);
            end
            
            % Calculate the gradient
            grads = dy * x_in.';
            
            % Back propogation
            dx = obj.backend.multiply_reverse( dy, obj.nlayer );
            dx = dx(1: obj.input_dim, :);
        end
    end
end