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

classdef model < handle
    properties 
        layer_list;
        backend;
        
        loss;
        optimizer;
        
        v;
    end
    
    methods
        function obj = model( backend )
            obj.layer_list = {};
            obj.backend = backend;
            
            obj.v = view();
        end
        
        function add( obj, layer, varargin)
            % Add a layer to the neural network model
            %
            
            % Calculate the input dimension according the output dimension
            % from last layer
            % INPUT
            %   layer:      The layer number
            %
            okargs = { 'net_corner' };
            defaults = {[1 1]};
            [net_corner ] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            
            if ~isempty( obj.layer_list )
                layer.input_dim = obj.layer_list{end}.output_dim;
            end
            
            if isnan( layer.input_dim)
                error('Must specify the input dimension for an input layer');
            end
            
            % set weight dimension
            layer.set_weight_dim();
            layer.backend = obj.backend;
            
            % Add one layer in the backend
            obj.backend.add_layer(layer.weight_dim, net_corner) % net_corner is only used in xbar backend
            
            % Add the current layer to the list
            layer.nlayer = numel(obj.layer_list) + 1;
            obj.layer_list{end+1} = layer;
        end
        
        function compile(obj, loss, optimizer, varargin )
            obj.loss = loss;
            
            optimizer.backend = obj.backend;
            obj.optimizer = optimizer;
            
            obj.backend.initialize_weights( varargin{:} );
        end
        
        function fit(obj, x_train, y_train, varargin )
            % FIT fit the model according to X_TRAIN and Y_TRAIN
            %   this is the main training function.
            % input: 
            %   x_train     Each element in the cell is one training sample
            %               x_train is a 2-dimensional or 3-dimensional
            %               array. x_train( :, n, t)
            %               Will need other structuring for a CNN in the
            %               future.
            %
            %   y_train     The corresponding expected outputs or labels.
            %   epochs
            %   batch_size
            %
            okargs = { 'batch_size', 'epochs'};
            defaults = {10, 1};
            [batch_size, epochs] = internal.stats.parseArgs(okargs, defaults, varargin{:});
     
            % 
            if size(x_train, 2) ~= size(y_train, 2)
                error('Training sample number mismatch!');
            end
            
            n_sample = size(x_train, 2);
            
            for ep = 1: epochs
                obj.v.print(['training on ep = ' num2str(ep)]);
                
                % process batch data
                for b_start = 1: batch_size: n_sample
                    b_end = min( b_start + batch_size - 1, n_sample);
                    
                    obj.v.print(['training on ' num2str(b_start) ' to ' num2str(b_end)]);
                    
                    % For simplicity, we only consider the case that all the time series 
                    % have the same time duration. 
                    % TODO: Test data with various time length.
                    %
                    x_batch = x_train(:, b_start: b_end, :);
                    y_batch = y_train(:, b_start: b_end, :);

                    loss_value = obj.fit_loop( x_batch, y_batch);
                    obj.v.plot(loss_value);
                end
            end
        end
        
        function y_test = predict(obj, x_test, varargin)
            okargs = { 'batch_size'};
            defaults = {10};
            [batch_size] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            
            y_test = [];
            n_sample = size(x_test, 2);
            
            for b_start = 1: batch_size: n_sample
                b_end = min( b_start + batch_size - 1, n_sample);

                x_batch = x_test(:, b_start: b_end, :);
                y_test = cat(2, y_test,  obj.forwardpass( x_batch ));
            end
        end
        
        function [accuracy, stats] = evaluate( ~, ys, ys_truth, varargin)
            okargs = { 'mode'};
            defaults = {'simple'};
            [eval_mode] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            [~, label_predict] = max(ys );
            [~, label_truth] = max(ys_truth );
            
            accuracy = mean(label_predict == label_truth);
            
            stats = struct();
            
            if strcmp( eval_mode, 'verbose')
                n_class = size(ys, 1);
                stats.win_count = zeros(n_class, n_class);
                
                for i = 1:size(label_predict, 2)
                    stats.win_count( label_truth(1,i, end), label_predict(1,i, end) ) = ...
                        stats.win_count( label_truth(1,i, end), label_predict(1,i, end) ) +1;
                end
                
                
            end
        end
    end
    
    methods (Access = private)
        function loss_value = fit_loop(obj, x_train, y_train)
            % FIT_LOOP an internal function to do the fitting (training)
            % INPUT
            %   x_train:    a two-dimensional array. Each column is one
            %               training sample.
            %   y_train:    Same as x_train.
            %
            ys   = obj.forwardpass( x_train);
            dys  = obj.loss.calc_delta( ys, y_train );
            loss_value = obj.loss.calc_loss( ys, y_train );
            
            grads = obj.backwardpass( dys );
            obj.optimizer.update( grads );
        end
        
        function y_ = forwardpass(obj, x_train)
            % FORWARDPASS the forward pass inferences through all the
            % layers.
            
            y_ = [];
            duration = size(x_train, 3);
            
            for t = 1:duration
                y_time = x_train(:,:,t);
                
                for l = 1:length( obj.layer_list )
                    if t == 1
                        % reset a recurrent layer when starts
                        obj.layer_list{l}.initialize();
                    end
                
                    y_time = obj.layer_list{l}.call( y_time );
                end
                
                y_ = cat(3, y_, y_time);
            end
        end
        
        function grads = backwardpass( obj, dys )
            % BACKWARDPASS calculate the weights gradients
            % INPUT
            %   dys:    final layer delta
            % OUTPUT
            %   grads:  Each element in the cell stores the cumulative
            %           weight gradient for one layer.
            %
            
            grads = cell(1, length( obj.layer_list) );
            grads(:) = {0};
            
            duration = size(dys, 3);
            
            % BP through time
            for t = duration: -1 :1
                % For layers
                dys_time = dys(:,:,t);
                for l = length( obj.layer_list ):-1 :1
                    [gradient, dys_time] = obj.layer_list{l}.calc_gradients(dys_time);
                    grads{l} = grads{l} + gradient;
                end
            end
        end
        
    end
end