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

classdef sim_array3 < memristor_array
    % Construction syntax: OBJ = SIM_ARRAY3(G0, RW, UPDATE_FUN, G_min, G_max)
    %
    % Notes: 
    % update_conductance change: I changed this from g=g+... to g=g-. I don't know the reason but it may work better...
    %
    %   -If you change any circuit parameters after construction,
    %   you'll get wrong answers unless you re-initialize a bunch of
    %   things. 
    %   -Keeps track of G_eff at all times, so reads are fast but updates
    %   are comparatively slow
    %   -Assumes voltages are applied from the right and currents are read
    %   from the bottom
    %
    %
    % TODO: Incorporate this with a realistic-device-behavior simulation.
    % TODO: Comment more
   
    properties
        net_size
        g_slow % Slow-read conductance
        g_eff % Effective conductance
        g_true % True value of conductance
        update_fun
        G_min
        G_max
        
        Gw
        s1 % 1st set of sparse indices
        s2
        Wm % Wire matrix
        Bm % Fixed B matrix
        
        col_select
        row_select
        
        R_wire % Wire resistance per segment (duplicate with Wm...)
        R_in % Wire resistance input
        R_out
        
    end
    methods
        %%
        function obj = sim_array3(g0, Rw, update_fun, G_min_input, G_max_input)
            % OBJ = SIM_ARRAY1(G0, RW, UPDATE_FUN, G_min, G_max)
            % creates a simulated memristor array.
            % G0 is the initial weight matrix, and determines the size of
            % the array. G0 can also be a cell array holding:
            %   {'random' SIZE MIN MAX}     Generates uniformly distributed
            %                               random weights between MIN and
            %                               MAX
            %   {'fixed' SIZE VALUE}        Initializes all weights to
            %                               exactly VALUE
            % UPDATE_FUN gives delta-G as a function of G, V_in, and V_trans
            % G_min defaults to 0
            % G_max defaults to Inf
            % The pulse width is fixed for now, but number of pulses may
            % eventually be something we can tune.
            
            switch nargin
                case {0 1 2}
                    error('Not enough input arguments')
                case 3
                    G_min_input = 100e-6;
                    G_max_input = 1000e-6;
                case 4
                    G_max_input = 1000e-6;
            end
            
            obj.G_min = G_min_input;
            obj.G_max = G_max_input;
            obj.g_slow = obj.generate_g0(g0);
            obj.g_true = obj.g_slow; %% I think need to remove accessing wire resistance...temporary value.
            obj.net_size = size(obj.g_slow);
            obj.row_select=1:obj.net_size(1);
            obj.col_select=1:obj.net_size(2);
            obj.update_fun = update_fun;
            obj.Gw = 1/Rw;
            obj.initialize_wires();
            obj.update_G_eff();
            
            obj.R_wire=1e-3;
            obj.R_in=1e-3;
            obj.R_out=1e-3;
        
            
        end
        %%
        function conductances = read_conductance(obj,varargin)
            % CONDUCTANCES = OBJ.READ_CONDUCTANCE() returns the conductance
            % values stored in the array.
            % TODO: Make it possible to fast and slow read.
            okargs = {'mode'};
            defaults = {'slow'};
            [mode,~,~] = internal.stats.parseArgs(okargs,defaults,varargin{:});
            
            if strcmpi(mode,'slow')
                conductances = obj.g_slow;
            elseif strcmpi(mode,'fast')
                conductances = obj.g_eff;
            end
        end
        %%
        function update_conductance(obj, V_col, V_row, V_gate)
            % OBJ.UPDATE_CONDUCTANCE(V_IN, V_OUT, V_TRANS)
            % Voltages are vectors, matrices, scalars. or 'GND'
            % They must match OBJ.NET_SIZE in at least one dimension if
            % they are not scalar.
            % One of V_OUT or V_IN should usually be 'GND'.
            %
            % This assumes that any update data we have is based on
            % slow-read values. 

            
            % Process inputs:
            V_col = obj.expand(V_col);
            V_row = obj.expand(V_row);
            V_gate = obj.expand(V_gate);
            
            if any(V_row(:)) && any(V_col(:))
                warning('Did you mean to SET and RESET both in one call?')
                % Two-vector SET is not supported for this class.
                % condition could be if any(V_out(:) & V_IN(:))
            end
            
            % Get conductance:
            g = obj.g_true(:);
            
            % Do the updates:
            if any(V_col(:))
                g = g+obj.update_fun(g,V_col(:),V_gate(:));
            end
            if any(V_row(:))
                g = g+obj.update_fun(g,-V_row(:),V_gate(:)); %% I changed this from g=g+... to g=g-. I don't know the reason but it may work better...
            end
            
            % Enforce the maxima:
            g(g<obj.G_min) = obj.G_min;
            g(g>obj.G_max) = obj.G_max;
            
            % Save result:
            obj.g_true(:) = g;
            
            % Recalculate effective conductances:
            M = obj.net_size(1); % Number of rows
            N = obj.net_size(2); % Number of columns
            obj.g_slow = 1./(1./obj.g_true + (M + 1 - (1:M)' + (1:N))/obj.Gw);
            obj.update_G_eff();
        end
        
        %%
        function I_out = read_current(obj, V_in, varargin)
            % Optional arguments can be entered but are ignored
            if iscell(V_in)
                try
                    V_in = cell2mat(V_in);
                catch
                    error('Unrecognized input format')
                end
            end
            
            I_out = obj.g_eff'*V_in;
        end
        
        %% Utilities
        %%
        function update_G_eff(obj)
            % Recompute effective conductances, assumming conductances are
            % up-to-date. 
            % Uses the logic of ResistorNetwork and the logic of fastread
            % together.
            
            M = obj.net_size(1); % Number of rows
            N = obj.net_size(2); % Number of columns
            
            % Construct matrix A:
            c = zeros(1,1,7); c(1:2) = 1; c(7) = -1; % Used to expand Gm
            m = obj.s1 > 0; % Mask for sparse indices
            vals = c .* obj.g_true + obj.Wm; % Entries of A
            A = sparse(obj.s1(m), obj.s2(m), vals(m), 2*M*N, 2*M*N);
            
            Vtb = -A\obj.Bm; % Calculate V top-to-bottom, for identity-matrix input
            % This line accounts for essentially all of the time used.
            
            indx1 = N*((1:M)'-1)+(1:N);
            indx2 = M*N+M*((1:N)-1)+(1:M)';
            Vd = Vtb(indx1,:)-Vtb(indx2,:); % Calculate V on each device            
            Id = Vd.*obj.g_slow(:); % Calculate current through each device
            
            obj.g_eff = reshape(sum(reshape(Id,M,N,M)),[N M])'; % And do a bunch of reshaping at the end
        end
        
        %%
        function initialize_wires(obj)
            % This code prepares to construct the A and B matrices.
            % TOO: Comment better.
            
            M = obj.net_size(1); % Number of rows
            N = obj.net_size(2); % Number of columns
            
            ivals = zeros(M,N,7);
            jvals = zeros(M,N,7);
            
            i = repmat((1:M)',1,N); % Define some index vectors
            j = repmat(1:N,M,1);
            a = N*(i-1)+j;
            b = M*N+M*(j-1)+i;
            
            ivals(:,:,1) = a;
            jvals(:,:,1) = a;
            
            ivals(:,:,2) = b;
            jvals(:,:,2) = b;
            
            ivals(:,:,3) = a.*(j ~= N); % 0 values are marked to discard
            jvals(:,:,3) = a+1;
            
            ivals(:,:,4) = a.*(j ~= 1);
            jvals(:,:,4) = a-1;
            
            ivals(:,:,5) = b.*(i ~= M);
            jvals(:,:,5) = b+1;
            
            ivals(:,:,6) = b.*(i ~= 1);
            jvals(:,:,6) = b-1;
            
            ivals(:,:,7) = b;
            jvals(:,:,7) = a;
            
            v0 = zeros(M,N,7);
            v0(:,:,1) = 2*obj.Gw;
            v0(:,:,2) = 2*obj.Gw;
            v0(:,:,3:6) = -obj.Gw;
            v0(:,:,7) = 0;
            
            v0(1:M,N,1) = v0(1:M,N,1)-obj.Gw; % Decrease certain diagonal elements
            v0(1,1:N,2) = v0(1,1:N,2)-obj.Gw;
            
            
            
            B = zeros(2*M*N, M);
            lin = sub2ind(size(B),N*((1:M)-1)+1,1:M); % Compute linear indices
            B(lin) = -obj.Gw;
            
            obj.s1 = ivals;
            obj.s2 = jvals;
            obj.Wm = v0;
            obj.Bm = B;
            
            % A = SPARSE(IVALS, JVALS, WM + MULT.*GM)
        end
        
        %%
        function b = expand(obj,a)
            % This function attempts to read minds: Did you mean _
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
        
%%
        function set_G(obj, G_input)
            % OBJ.SET_G(obj, g)
            % change g_true (slow read) to input g matrix
            % for debugging/other purposes...
            
            % Dimension check
            if ~isequal(size(G_input),size(obj.g_true))
                error('Input matrix size mismatches the memristor size.');
            end
            
            obj.g_true(:) = G_input;
            
            % Recalculate effective conductances:
            M = obj.net_size(1); % Number of rows
            N = obj.net_size(2); % Number of columns
            obj.g_slow = 1./(1./obj.g_true + (M + 1 - (1:M)' + (1:N))/obj.Gw);
            obj.update_G_eff();
        end
    end
end