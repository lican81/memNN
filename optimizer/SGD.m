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

classdef SGD < optimizer
    properties
        lr = 0;
        momentum = 0;
        
        backend;
        dWs_pre;
    end
    
    methods
        function obj = SGD( varargin )
            % SGD constructor function for stochasitc gradient descent with
            % momentum. Will add learning rate decay later.
            %
            okargs = {'lr', 'momentum'};
            defaults = {0.1, 0};
            [obj.lr, obj.momentum] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
%             obj.backend = [];
            obj.dWs_pre = {};
        end
        
        function update(obj, grads)
            % Enforce the learning rate
            % TODO: check momentum, it is NOT working yet!!
            %
            dWs = cellfun(@(x) x.*obj.lr, grads,'UniformOutput',false);
            
            if ~isempty( obj.dWs_pre)
                for l = 1: length(grads)
                    % Add momentum
                    dWs{l} = dWs{l} + obj.dWs_pre{l} .* obj.momentum;
                end
            end
            
            % hardware call
            obj.backend.update( dWs );
            
            % store the weight updates for future use
            obj.dWs_pre = dWs;
        end
            
    end
end