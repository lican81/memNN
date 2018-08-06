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

% This class maps the numerical values to physical ones to interface with
% sim_array or real_array.

classdef software < handle
    properties
        %store the weights
        W
    end
    methods
        function obj = software( )
            obj.W = {};
        end
        
        function add_layer(obj, weight_dim, varargin)
            % ADD_LAYER add another layer to the software backend.
            %
            %
            
            % Add one layer
            obj.W{end+1} = 2 * rand( weight_dim ) - 1;
            
            % Normalization
            obj.W{end} = obj.W{end} / sqrt( size(obj.W{end}, 2) );
        end
        
        function initialize_weights(~, varargin)
        end
        
        function update(obj, dW)
            %UPDATE update weights by layers
            %
            for l = 1:numel( dW )
                obj.W{l} = obj.W{l} + dW{l};
            end
        end
        
        function check_layer(obj, layer )
            if layer > numel(obj.W)
                error(['layer number should be less than ' num2str(numel(obj.W))]);
            end
        end
        
        function output = multiply(obj, vec, layer)
            obj.check_layer(layer);
            
            output = obj.W{layer} * vec;
        end
        
        function output = multiply_reverse(obj, vec, layer)
            obj.check_layer(layer);
            
            output = obj.W{layer}.' * vec;
        end
    end
end