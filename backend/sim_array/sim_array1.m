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

classdef sim_array1 < memristor_array
    % Simulated subarray
    % Assumes 0 wire resistance, all devices identical, 100% yield, 0 noise
    % Construction syntax: OBJ = SIM_ARRAY1(G0, UPDATE_FUN, GMIN, GMAX)
    properties
        net_size
        conductances
        update_fun
        gmin
        gmax
    end
    methods
        %%
        function obj = sim_array1(g0, update_fun, gmin, gmax)
            % OBJ = SIM_ARRAY1(G0, UPDATE_FUN, GMIN, GMAX)
            % creates a simulated memristor array.
            % G0 is the initial weight matrix, and determines the size of
            % the array. G0 can also be a cell array holding:
            %   {'random' SIZE MIN MAX}     Generates uniformly distributed
            %                               random weights between MIN and
            %                               MAX
            %   {'fixed' SIZE VALUE}        Initializes all weights to
            %                               exactly VALUE
            % UPDATE_FUN gives delta-G as a function of G, V_in, and V_trans
            % GMIN defaults to 0
            % GMAX defaults to Inf
            % The pulse width is fixed for now, but number of pulses may
            % eventually be something we can tune.
            
            switch nargin
                case {0 1}
                    error('Not enough input arguments')
                case 2
                    gmin = 0;
                    gmax = Inf;
                case 3
                    gmax = Inf;
            end
            
            obj.gmin = gmin;
            obj.gmax = gmax;
            obj.conductances = obj.generate_g0(g0);
            obj.net_size = size(obj.conductances);
            obj.update_fun = update_fun;
        end
        %%
        function conductances = read_conductance(obj,varargin)
            % CONDUCTANCES = OBJ.READ_CONDUCTANCE() returns the conductance
            % values stored in the array. This is noise-free reading.
            % Optional input arguments can be entered but are ignored.
            conductances = obj.conductances;
        end
        %%
        function update_conductance(obj, V_in, V_out, V_trans)
            % OBJ.UPDATE_CONDUCTANCE(V_IN, V_OUT, V_TRANS)
            % V_in and V_trans are vectors, matrices, or scalars
            % V_out is fixed grounded right now.
            % No need to return the object
            
            % In the actual array, I think there might be a fast way to do many
            % conductance updates - later.
            %
            % TODO: For cell inputs, call repeatedly. Or, better, make a
            % "sequence" function, which tracks the state after each call. 
            
            % Process inputs:
            V_in = obj.expand(V_in);
            V_out = obj.expand(V_out);
            V_trans = obj.expand(V_trans);
            
            if any(V_out(:)) && any(V_in(:))
                warning('Did you mean to SET and RESET both in one call?')
                % Two-vector SET is not supported for this class.
                % condition could be if any(V_out(:) & V_IN(:))
            end
            
            % Get conductance:
            g = obj.conductances(:);
            
            % Do the updates:
            if any(V_in(:))
                g = g+obj.update_fun(g,V_in(:),V_trans(:));
            end
            if any(V_out(:))
                g = g+obj.update_fun(g,-V_out(:),V_trans(:));
            end
            
            % Enforce the maxima:
            g(g<obj.gmin) = obj.gmin;
            g(g>obj.gmax) = obj.gmax;
            
            % Save result:
            obj.conductances(:) = g;
        end

%%        
        function I_out = read_current(obj, V_in, varargin)
            % Optional arguments can be entered but are ignored
            if iscell(V_in)
                try
                    cell2mat(V_in);
                catch
                    error('Unrecognized input format')
                end
            end
            
            I_out = obj.conductances'*V_in;
        end
        
        % Other methods to write: Some sort of testing, probably.
        % Maybe also a "set-weights" type of thing, for initialization
        
        %%
        function b = expand(obj,a)
            % this function attempts to read minds: Did you mean _
            % Does its best to expand things.
            if all(size(a) == obj.net_size)
                b = a;
            elseif isscalar(a)
                b = repmat(a,obj.net_size);
            elseif strcmpi(a,'GND')
                b = zeros(obj.net_size);
            elseif any(size(a) == obj.net_size)
                if iscolumn(a) && length(a) == obj.net_size(1)
                    b = repmat(a,1,obj.net_size(2));
                elseif isrow(a) && length(a) == obj.net_size(2)
                    b = repmat(a,obj.net_size(1),2);
                else
                    error('Not sure how to expand this input')
                end
            else
                error('Not sure how to expand this input')
            end
        end
        %%
        function g0 = generate_g0(~,g0)
            % Used to automatically generate initial weight matrices.
            % I'll add more options to them as they become useful.
            if isnumeric(g0)
                return
            elseif iscell(g0) % Case where it's a cell
                switch lower(g0{1})
                    case 'random'
                        % Second entry is size, third is minval, 4th is
                        % maxval.
                        g0 = rand(g0{2})*(g0{4}-g0{3})+g0{3};
                    case 'fixed'
                        % Second entry is size, third is value
                        g0 = zeros(g0{2})+g0{3};
                end
            end
        end
    end
end