% Copyright (c) 2018, 
% Department of Electrical and Computer Engineering, University of Massachusetts Amherst
% All rights reserved.
% 
% Created by 
%     Zhongrui Wang,
%     Can Li (Email: me at ilican dot com
%             Website: http://ilican.com)
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

classdef RMSprop < optimizer
    properties
        
        % RMSprop parameters
        lr = 0;
        momentum = 0;
        decay = 0;
        eps = 0;
        
        backend;
        
        % History of previous dWs and gradient mean square
        dWs_pre;
        grad_mean_sqr
    end
    
    methods
        function obj = RMSprop( varargin )
            % RMSprop constructor function for RMSprop with
            % momentum. 

            okargs = {'lr', 'momentum', 'decay', 'eps'};
            defaults = {0.001, 0, 0.9, 1e-8};
            [obj.lr, obj.momentum, obj.decay, obj.eps] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
%             obj.backend = [];
            obj.dWs_pre = {};
            obj.grad_mean_sqr = {};
        end
        function update(obj, grads)
            % Input grads: cell array of gradients, each layer per cell
            
            % If it's the first time to use grad_mean_sqr
            if isempty(obj.grad_mean_sqr)
                obj.grad_mean_sqr = cellfun(@(x) zeros(size(x)), grads,'UniformOutput',false);
            end
                
            % If it's the first time to use momentum
            if isempty(obj.dWs_pre)                
                obj.dWs_pre = cellfun(@(x) zeros(size(x)), grads,'UniformOutput',false);
            end
            
            % For all layers
            for l = 1: length(grads)
                
                % Gradient mean square (evolution)
                obj.grad_mean_sqr{l}=obj.decay*obj.grad_mean_sqr{l}+...
                    (1-obj.decay)*grads{l}.^2;
        
                % Add momentum
                obj.dWs_pre{l}= obj.lr * grads{l}./(obj.grad_mean_sqr{l}.^0.5+obj.eps)...
                        -obj.momentum * obj.dWs_pre{l};
                    
            end
            
            % hardware call (although it's dWs_pre, it's not pre at this
            % moment)
            obj.backend.update( obj.dWs_pre );
            
        end
    end
end