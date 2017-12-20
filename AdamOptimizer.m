classdef AdamOptimizer < handle
    
    properties
        step_size
        beta1
        beta2
        epsilon

        x
        m
        v
        cnt
    end

    methods

        function obj = AdamOptimizer(x0, step_size, beta1, beta2, epsilon)
            obj.step_size = 1e-3;
            obj.beta1 = 0.9;
            obj.beta2 = 0.999;
            obj.epsilon = sqrt(eps);

            n_params = numel(x0);
            obj.x = x0(:);
            obj.m = zeros(n_params, 1);
            obj.v = zeros(n_params, 1);
            obj.cnt = 0;
        end

        function step(obj, sg)
            obj.cnt = obj.cnt + 1;

            % update biased 1st moment estimate
            obj.m = obj.beta1 .* obj.m + (1 - obj.beta1) .* sg(:);
            % update biased 2nd raw moment estimate
            obj.v = obj.beta2 .* obj.v + (1 - obj.beta2) .* (sg(:).^2);
            
            % bias-corrected 1st moment estimate
            mHat = obj.m  ./ (1 - obj.beta1^obj.cnt);
            % bias-corrected 2nd raw moment estimate
            vHat = obj.v ./ (1 - obj.beta2^obj.cnt);
            
            % update parameters
            obj.x = obj.x - obj.step_size .* mHat ./ (sqrt(vHat) + obj.epsilon);
        end

    end

end