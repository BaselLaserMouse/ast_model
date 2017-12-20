classdef AdamOptimizer < handle
    % ADAM optimizer class
    % based on https://github.com/mp4096/adawhatever implementation
    %
    % SEE ALSO AdamOptimizer/AdamOptimizer

    properties
        step_size  % gradient descent learning rate
        beta1      % 1st moment decay rate
        beta2      % 2nd moment decay rate
        epsilon    % constant for numerical stability
    end

    properties (SetAccess = private)
        cnt  % iterations counter
        x    % optimized parametes
        m    % estimate of 1st moment (mean) of the gradient
        v    % estimate of 2nd moment (uncentered variance) of the gradient
    end

    methods

        function obj = AdamOptimizer(x0, varargin)
            % AdamOptimizer constructor

            parser = inputParser;
            parser.addParameter('step_size', 1e-3);
            parser.addParameter('beta1', 0.9);
            parser.addParameter('beta2', 0.999);
            parser.addParameter('epsilon', sqrt(eps));

            parser.parse(varargin{:});
            obj.step_size = parser.Results.step_size;
            obj.beta1 = parser.Results.beta1;
            obj.beta2 = parser.Results.beta2;
            obj.epsilon = parser.Results.epsilon;

            n_params = numel(x0);
            obj.cnt = 0;
            obj.x = x0(:);
            obj.m = zeros(n_params, 1);
            obj.v = zeros(n_params, 1);
        end

        function step(obj, sg)
            % one step of Adam optimization

            obj.cnt = obj.cnt + 1;

            % update biased moment estimates
            obj.m = obj.beta1 .* obj.m + (1 - obj.beta1) .* sg(:);
            obj.v = obj.beta2 .* obj.v + (1 - obj.beta2) .* (sg(:).^2);

            % bias-corrected moment estimates
            mHat = obj.m  ./ (1 - obj.beta1^obj.cnt);
            vHat = obj.v ./ (1 - obj.beta2^obj.cnt);

            % update parameters
            obj.x = obj.x - obj.step_size .* mHat ./ (sqrt(vHat) + obj.epsilon);
        end

    end

end