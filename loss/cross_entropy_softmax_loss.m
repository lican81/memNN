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

classdef cross_entropy_softmax_loss < loss
    % Cross-entropy loss for softmax  
    properties
        calc_mode
        recurrent_mode
    end
    
    methods
        function obj = cross_entropy_softmax_loss( varargin )
            obj = obj@loss( varargin{:} );
        end
        
        function dys  = calc_delta(obj, ys, ys_train )
            dys = ys_train - ys;
            
            % Only the last output matters for some recurrent neural networks
            if strcmp( obj.recurrent_mode, 'last')
                dys(:,:,1:end-1) = 0;
            end
        end
          
        function loss = calc_loss(obj, ys, ys_train )
            loss = 0;
            
            for t = 1: size(ys, 3)
                %Only the last output matters for some recurrent neural networks
                if strcmp( obj.recurrent_mode, 'last') && t ~= size(ys, 3)
                    continue;
                end
            
                for n = 1: size(ys, 2)
                    for c = 1: size(ys, 1)
                        loss = loss - ys_train( c, n, t) * log( ys(c, n, t) );
                    end
                end
            end
            
            if strcmp( obj.calc_mode, 'mean')
                loss = loss ./  size(ys, 2) ./ size(ys, 3);
            end
        end
    end
end