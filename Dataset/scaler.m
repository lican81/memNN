% Scaler to normalzie data
% MinMaxScaler for now.
%
% Scales the data by output = input * \alpha + \beta

classdef scaler < handle
    properties
        alpha
        beta
        
        range_output;
    end
    methods
        function obj = scaler( range )
            obj.range_output = range;
        end
        
        function output = scale( obj, input )
            obj.alpha = (obj.range_output(2) - obj.range_output(1)) / ( max( input(:)) - min(input(:)) );
            obj.beta = - min( input(:) ) * obj.alpha + obj.range_output(1);
            
            output = obj.alpha * input + obj.beta;
        end
        
        function output = recover( obj, input )
            output = ( input - obj.beta ) / obj.alpha;
        end
    end
end