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

classdef real_array1 < memristor_array
    % This class interacts with the crossbar through DPE_WRITING(). It uses
    % only a subsection of the array, and devices outside that subsection
    % are unaffected. This class is intended to be fairly lightweight, with few built-in
    % functions, but it a wide range of algorithms can be written on top of
    % it. 
    % Construction syntax: 
    % OBJ = REAL_ARRAY1(NET_SIZE, NET_CORNER = [1 1],ARRAY_SIZE = [128 64])
    
    % TODO: add strict maxima on voltages
    properties
        net_size
        array
        array_size
        net_corner
        mask
        
    end
    methods
%%
        function obj = real_array1(net_size,net_corner,array_size)
            % OBJ = REAL_ARRAY1(NET_SIZE) creates an array object for a 128x64
            % crossbar using a subsection in the upper left. This class interacts
            % with the crossbar through DPE_WRITING.
            % The subsection used is NET_SIZE(1) by NET_SIZE(2), and no device
            % outside that subsection is affected.
            % OBJ = REAL_ARRAY1(NET_SIZE,NET_CORNER,ARRAY_SIZE,) uses the subsection
            % specified by NET_CORNER for a crossbar of size ARRAY_SIZE.
    
            % Size is n_in by n_out
            nc = [1 1]; as = [128 64]; % defaults
            switch nargin
                case 0
                    error('Network size must be specified')
                case 2
                    nc = net_corner;
                case 3
                    nc = net_corner;
                    as = array_size;
            end
            
            if any(net_size+nc-1>as)
                error('Network exceeds array bounds')
            end
            
            obj.net_size = net_size; 
            obj.array_size = as;
            obj.net_corner = nc;
            
            % Create a logical mask showing which elements are part of the
            % net:
            obj.mask = false(as);
            obj.mask(nc(1):nc(1)+net_size(1)-1, nc(2):nc(2)+net_size(2)-1) = true;
            
            % Initialize array connection:
            obj.array = dpe_writing();
            obj.array.connect();
        end
%%        
        function conductances = read_conductance(obj, varargin)
            % G = OBJ.READ_CONDUCTANCE() reads all conductances in the
            % array and returns them as a matrix.
            % G = OBJ.READ_CONDUCTANCE('NAME',VALUE) controls some
            % parameters of the read operation. Optional arguments and
            % default values are:
            %   'v_read' = 0.2
            %   'gain' = 2
            %   'mode' = 'slow'
            if ~any(cellfun(@(x)strcmpi(x,'subarray'),varargin)) % If subarray is unset
                s = 'subarray'; % Used inside evalc
                [~,conductances] = evalc('obj.array.read(varargin{:},s,[obj.net_corner obj.net_size])'); % HARDWARE CALL
            else
                [~,conductances] = evalc('obj.array.read(varargin{:})');
            end
            if any(size(conductances) ~= obj.net_size)
                conductances = reshape(conductances(obj.mask),obj.net_size);
            end
        end
%%        
        function update_conductance(obj, V_in, V_out, V_trans)
            % OBJ.UPDATE_CONDUCTANCE(V_IN,V_OUT,V_TRANS) sends and update
            % pulse to the array.
            % All voltages can be 'GND', a scalar, a column vector of length
            % net_size(1), or a matrix of size net_size.
            % Best practice is to set one of V_IN or V_OUT to 'GND'. If
            % both are nonzero, then two separate pulses (one SET and one
            % RESET) will be applied.
            
            V_gate = obj.expand(V_trans);
            V_in = obj.expand(V_in); % Voltage applied to each column ( Should be *V_set?)
            V_out = obj.expand(V_out); % Voltage applied to each row on a subsequent pulse
            
            % Only handles nonnegative voltages
            
            if any(V_in(:)>0) && any(V_out(:)>0)
                warning('Did you mean to set and reset all in one call?')
            end

            if any(V_in(:)>0)
                %obj.array.batch_set(V_in,V_gate,'print',false); % HARDWARE CALL
                [~] = evalc('obj.array.batch_set(V_in, V_gate)');
            end
            if any(V_out(:)>0)
                %obj.array.batch_reset(V_out, V_gate,'print',false); % HARDWARE CALL
                [~] = evalc('obj.array.batch_reset(V_out, V_gate)');
            end
        end
%%        
        function I_out = read_current(obj,V_in, varargin)
            % I_OUT = READ_CURRENT(V_IN) does matrix-matrix multiplication.
            % I_OUT = transpose(G)*V_IN when G is 128x64. Note that this
            % function interprets the orientations of I_OUT and V_in
            % differently than VMM_hardware.
            % V_IN can be either a row cell array of column vectors or a matrix.
            % In either case, I_OUT will be a matrix.
            % Optional args:
            %   gain = 1
            
            if iscell(V_in)
                try
                    cell2mat(V_in);
                catch
                    error('Unrecognized input format')
                end
            end
            
            
            temp = padarray(V_in,obj.net_corner(1)-1,0,'pre'); % throwaway
            V_in = padarray(temp,obj.array_size(1)-obj.net_corner(1)-obj.net_size(1)+1,0,'post')';
            
            %[~,I_out] = evalc('obj.array.VMM_hardware(V_in, varargin{:})'); %HARDWARE CALL
            I_out = obj.array.VMM_hardware(V_in, varargin{:}); %HARDWARE CALL
            
            % TODO: I am thinking if we need to use the dynamci gain, the
            % activation will be sort of segamentaed linear relation.
            %
            % The following code will do it. Will consider how to pass the
            % varagin properly.
            %
            th_gain2 = 2.2e-4;
            I_out_gain1 = obj.array.VMM_hardware(V_in, 'gain', 1);
            I_out_gain2 = obj.array.VMM_hardware(V_in, 'gain', 2);
            
            I_out = I_out_gain1;
            I_out( abs(I_out) < th_gain2 ) = I_out_gain2( abs(I_out) < th_gain2 );
            
            
            I_out = I_out(:,obj.net_corner(2):obj.net_corner(2)+obj.net_size(2)-1);
            I_out = I_out';
        end
          
%%        
        function c = expand(obj,a,x)
            % B = OBJ.EXPAND(A) zero-pads array A so that it lines up right
            % with the object matrix.
            % B = OBJ.EXPAND(A,X), where X is a scalar, pads with X
            % instead.
            % This is a utility designed for use inside of other methods on
            % this class.
            
            % First part: Expand a to net_size
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
            
                        
            % Second part: Pad b to array_size
            if nargin<3
                x = 0;
            end
            
            c = zeros(obj.array_size)+x;
            c(obj.mask) = b;
        end
        
        %%
        function reconnect(obj)
%             obj.array.disconnect();
            obj.array.connect();
        end
    end
end
     
        
% I may want functions to:
%   Initialize all conductances to some desired value
%   Bring a subset of conductances to some desired value
%   Read a subset of conductances (definitely want this, TODO)
% 

% It may eventually be useful to rewrite this so that it interacts
% more directly with the device, instead of going through these layers of
% abstraction... consider it.