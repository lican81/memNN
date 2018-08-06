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

%% Prepare the dataset
% The input data for the LSTM should follow 
% xs = [data_dimension, n_sample, time_step ]
% 
% For MNIST, we use the image columns as inputs, and each input is one time
% step. LSTM is used as a feature extractor across time steps (for MNIST,
% it is actually spatial feature)
%
initialize;

load('mnist_training_8x8.mat');

xs_trainig = reshape( data, 8, 8, []);
xs_trainig = permute( xs_trainig, [1 3 2]);

n = length(labels);
ys_training = zeros( 10, n, 8);
for i = 1:n
    ys_training( labels(i) + 1, i, : ) = 1;
end

load('mnist_test_8x8.mat');

xs_testing = reshape( data, 8, 8, []);
xs_testing = permute( xs_testing, [1 3 2]);

n = length(labels);
ys_testing = zeros( 10, n, 8);
for i = 1:n
    ys_testing( labels(i) + 1, i, : ) = 1;
end

%% Neural network
m = model( software() );

m.add( LSTM(54, 'input_dim', 8 ) );
m.add( dense(10, 'activation', 'stable_softmax')  );
          
m.compile(cross_entropy_softmax_loss('recurrent_mode', 'last'),...
    RMSprop('lr', 0.01 ) );
          
%
m.v.DRAW_PLOT = 0;
m.v.DEBUG = 0;

for i = 1:20
    m.fit( xs_trainig, ys_training, 'batch_size', 50, 'epochs', 1);
    ys = m.predict(xs_testing, 'batch_size', 50);
    
    accuracy = m.evaluate( ys, ys_testing);
    
    disp(['i=' num2str(i)...
        ' accuracy=' num2str( accuracy )]);
end
