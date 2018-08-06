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

%% Prepare for the dataset
close all;
clear;
initialize;
load('gait_subset.mat');

%%
% noise_list = [0.0001 0.005 0.01 0.015 0.02 0.025 0.03];
% noise_list = 0:0.002:0.03;
% noise_list(1) = 0.0001;

noise_list = 0.014;

for iiii = 1: length(noise_list)
    noise = noise_list(iiii);

    f = @(Vt) 98e-6*Vt + 120e-6;

    update_fun = @(G,V,Vt) -G.*(V ~= 0)+(V > 0).* max(f(Vt),G) + noise.*randn(size(G)).* 400e-6; 



    acc_sim2 = {};
    loss_sim2 = {};

    parfor repeat = 1:50

        base = multi_array(sim_array1({'random' [128 64] 50e-6 100e-6},update_fun,0, Inf));
        m = model( xbar( base ) );


        m.add( LSTM(14, 'input_dim', 50 , 'bias_config', [0 0]) );
        m.add( dense(N_CLASS, 'activation', 'stable_softmax', 'bias_config', [0 0]), ...
            'net_corner', [1 57]);

        m.compile(cross_entropy_softmax_loss('recurrent_mode', 'last', 'calc_mode', 'mean'),...
            RMSprop('lr', 0.01, 'momentum', 0.0 ) );

        %
        m.v.DRAW_PLOT = 0;
        m.v.DEBUG = 0;
        m.backend.draw = 0;
        m.backend.ratio_G_W = 100e-6;

    %
        acc = [];
        for i = 1:50
            tt = tic;

            m.fit( xs_training, ys_training, 'batch_size', 50, 'epochs', 1);
            ys = m.predict(xs_testing, 'batch_size', 50);

            accuracy = m.evaluate( ys, ys_testing);

            tElapsed = toc(tt);
            disp(['i=' num2str(i)...
                ' accuracy=' num2str( accuracy(:,:,end) ) '; time elapsed in the cycle=' num2str(tElapsed) 's' ]);

            acc = [acc accuracy];
        end
        acc_sim2{repeat} = acc;
        loss_sim2{repeat} = m.v.data_plot;
    end
    save(['acc_testset_simu_noise_' num2str(noise) '.mat'], 'acc_sim2', 'loss_sim2');
end

[accuracy, stats] = m.evaluate( ys, ys_testing, 'mode', 'verbose');