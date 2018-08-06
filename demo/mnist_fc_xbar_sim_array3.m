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
initialize;
load('mnist_training_8x8.mat');

xs_trainig = data;

n = length(labels);
ys_training = zeros( 10, n);
for i = 1:n
    ys_training( labels(i) + 1, i ) = 1;
end

load('mnist_test_8x8.mat');

xs_test = data;

n = length(labels);
ys_test = zeros( 10, n);
for i = 1:n
    ys_test( labels(i) + 1, i ) = 1;
end


%%
f = @(Vt) 98e-6*Vt + 120e-6;
noise = 0.01;
update_fun = @(G,V,Vt) -G.*(V ~= 0)+(V > 0).* max(f(Vt),G) + noise.*randn(size(G)).*400e-6; 

base = multi_array(sim_array3({'random' [128 64] 50e-6 100e-6}, 0.1, update_fun,0, Inf));


m = model( xbar( base ) );
% m = model( software( ) );

m.add( dense(54, 'input_dim', 64, 'activation', 'relu', 'bias_config', [0 0]));
m.add( dense(10, 'activation', 'softmax', 'bias_config', [0 0]), ...
    'net_corner', [1 55] );

m.compile(cross_entropy_softmax_loss(),...
              RMSprop('lr', 0.002, 'momentum', 0.0) );
          

%%
m.v.DRAW_PLOT = 1;
m.v.DEBUG = 1;

for i = 1:20
    m.fit( xs_trainig, ys_training, 'batch_size', 50, 'epochs', 1);
    ys = m.predict(data, 'batch_size', 50);

    accuracy = m.evaluate( ys, ys_test);
    
    disp(['i=' num2str(i)...
        ' accuracy=' num2str( accuracy )]);
end

