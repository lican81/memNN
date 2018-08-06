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

classdef activations
    % This class provides various activation functions

    methods(Static)
        function act = get( act_name, direction)
            % Get the activation function handle. 
            % Input:
            %   act_name:   The name of the activation function, in string
            %   direction:  'act' | 'deriv' 
            %               The function itself | Its derivative
            % Output:
            %   The function handle
            
%             disp([ act_name, '', direction]);
            if strcmp(direction, 'deriv')
                d_str = 'd';
            else
                d_str = '';
            end

            act = str2func(['activations.' d_str act_name]);
        end
        
        % Activation functions
        function y = linear(x)
            y = x;
        end
        
        function y = dlinear(~)
            y = 1;
        end
        
        function y = tanh(x)
            y = tanh(x);
        end
        
        function y = dtanh(x)
            y = 1-x.^2;
        end
        
        function y = sigmoid(x)
            y = 1./(1 + exp(-x));
        end
        
        function y = dsigmoid(x)
            y = x.*(1 - x);
        end
        
        function y = relu(x)
            y = max(0, x);
        end
        
        function y = drelu(x)
            y = x > 0;
        end
        
        function y = softmax(x)
            y = exp(x) ./ sum( exp(x) );
        end
        
        function y = stable_softmax(x)
            y=exp(x-max(x)) ./ sum(exp(x-max(x)));
        end
    end
end