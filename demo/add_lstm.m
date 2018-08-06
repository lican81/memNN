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

dim_binary = 8;
n_sample = 11000;

xs = [];
ys = [];

disp('generating numbers...');
for n = 1: n_sample
    largest_number = 2^dim_binary;

    a_int = randi(largest_number /2);
    b_int = randi(largest_number /2);
    
    a = double( dec2bin(a_int, dim_binary) ) -'0';
    b = double( dec2bin(b_int, dim_binary) ) -'0';
    
    in = [a;b];
    
    in = in(:, end:-1:1);
    
    % ground true
    c_int = a_int + b_int;
    out = double( dec2bin(c_int, dim_binary) ) - '0';
    out = out(:, end:-1:1);
    
    xs = cat(2, xs, permute( in, [1 3 2]) );
    ys = cat(2, ys, permute(out, [1 3 2]) );
end
disp('done');

xs_training = xs(:,1:10000, :);
ys_training = ys(:,1:10000, :);

%% Network training with framework

m = model( software() );

m.add( LSTM(12, 'input_dim', 2) );
m.add( dense(1, 'activation', 'sigmoid')  );

m.compile(mean_square_loss('calc_mode', 'sum'),...
    RMSprop('lr', 0.05, 'momentum', 0.0) );

m.v.DRAW_PLOT = 1;
m.v.DEBUG = 0;

% for i = 1:1000
m.fit( xs_training, ys_training, 'batch_size', 10, 'epochs', 1);
% y_predit = m.predict(xs_testing, 'batch_size', 1);




