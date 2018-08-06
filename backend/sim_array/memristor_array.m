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

classdef (Abstract) memristor_array < handle
    % A superclass for other array interfaces
    % The purpose of this is just to be able to test identical training
    % protocols on real and simulated arrays.
    % Additionally, could be applied to simulations of various styles, or
    % various types of memristors, etc
    
    properties (Abstract)
        net_size % The only property common to all memristor_arrays
    end
    methods (Abstract)
        conductances = read_conductance(obj, varargin)
        update_conductance(obj, V_in, V_out, V_gate)            
        I_out = read_current(obj, V_in)
    end
    methods
        function [exitflag,stats] = set(obj,G_target,varargin)
            [exitflag,stats] = set_conductance(obj,G_target,varargin{:});
        end
        function [G,stats] = effective_read(obj,varargin)
            if nargout>1
                [G,stats] = effective_conductance(obj,varargin{:});
            else
                G = effective_conductance(obj,varargin{:});
            end
        end
    end
end


% Goal: Separate training protocol and hardware/simware
% Training is handled by "perceptron" or "layer" class
% Communication (expansion, calculation of v_trans) and/or simulation is
% handled by "array" class
