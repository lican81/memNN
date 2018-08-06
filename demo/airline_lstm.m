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

%% DATA
initialize;
load('airline.mat');


num_passengers = cell2mat( internationalairlinepassengers2(:, 2) );

s = scaler( [0 1] );
num_passengers = s.scale(num_passengers);
%

n_window = 1;
data = [];
for i = 1: n_window
    data = [data; num_passengers(i:end-n_window -1 + i)'];
end
    
% data = num_passengers(1:end-1)';
ys = num_passengers(n_window + 1:end)';

num_training = floor( size(data, 2) *2 /3);

xs_training = data(:, 1: num_training);
ys_training = ys(:, 1: num_training);

xs_testing = data; %( :, num_training + 1: size(data, 2) );
ys_testing = ys; %( :, num_training + 1: size(data, 2) );

%
xs_training = permute(xs_training, [1 3 2]);
ys_training = permute(ys_training, [1 3 2]);
xs_testing = permute(xs_testing, [1 3 2]);
ys_testing = permute(ys_testing, [1 3 2]);

%% Network training with framework
%%
f = @(Vt) 98e-6*Vt + 120e-6;
noise = 0.01;
update_fun = @(G,V,Vt) -G.*(V ~= 0)+(V > 0).* max(f(Vt),G) + noise.*randn(size(G)).* 400e-6; 

base = multi_array(sim_array1({'random' [128 64] 50e-6 100e-6},update_fun,0, Inf));

m = model( xbar( base ) );
%

% m = model( software() );

m.add( LSTM(15, 'input_dim', 1, 'bias_config', [0.2 1]) );
m.add( dense(1, 'activation', 'sigmoid', 'bias_config', [0.2 1]), 'net_corner', [1 61]  );

m.compile(mean_square_loss('calc_mode', 'sum'),...
    SGD('lr', 0.01, 'momentum', 0.9) );

m.v.DRAW_PLOT = 1;
m.v.DEBUG = 0;

% for i = 1:1000
m.fit( xs_training, ys_training, 'batch_size', 1, 'epochs', 800);
y_predit = m.predict(xs_testing, 'batch_size', 1);


%%
figure(2);
clf;
plot(s.recover(reshape(y_predit, 1, []) ) );
hold on;
plot(s.recover(reshape(ys_testing, 1, [])), '--');

ylabel('Number of passengers');

xticks(1:12:144);
xtickyears = cellfun(@(x) char(x), internationalairlinepassengers2(1:12:144, 1) ,'UniformOutput',false);
xtickyears = cellfun(@(x) x(3:4), xtickyears ,'UniformOutput',false);

xticklabels(xtickyears );
grid on;


