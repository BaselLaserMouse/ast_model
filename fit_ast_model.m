function [cleaned_trace, var_params, v_elbos] = fit_ast_model(traces, n_sectors, varargin)
    % FIT_AST_MODEL fit an Asymmetric Student-t model to remove neuropil data
    %
    % [cleaned_trace, var_params, v_elbos] = fit_ast_model(traces, n_sectors, ...)
    %
    % INPUTS
    %   traces - signals from ROI and neighborhood, as a [2 Time] array
    %   n_sectors - number of pixels for each signal, as a 2-elements vectors
    %
    % NAME-VALUE PAIR INPUTS (optional)
    %   n_dct - default: 1
    %       number of DCT basis functions to use to approximate trends
    %   n_samples - default: 1
    %       number of samples used for black-box variational inference
    %   maxiter - default: 10000
    %       max. number of iterations for Adam based stochastic gradient descent
    %   adam_step - default: 1e-2
    %       step size for Adam optimizer
    %   tolerance - default: 1e-4
    %       relative change in averaged ELBO to stop optimization
    %   winsize - default: 200
    %       number of consecutive ELBO estimates to average (cf. tolerance)
    %   verbose - default: false
    %       level of verbosity as either
    %       1) false or 0: disabled
    %       2) true or 1: print messages
    %       3) 2 or more: plot results
    %
    % OUTPUTS
    %   cleaned_trace - decontaminated signal, as a [Time] vector
    %   var_params - optimized variational parameters, as vector concatenating
    %       mean and variance of Gaussian posteriors
    %   v_elbos - values of ELBO cost function, as a [maxiter] vector
    %
    % SEE ALSO AdamOptimizer

    % TODO meaningful priors (mu_mean == 1 ?, informed prior for coeff?)
    % TODO meaningful initial values

    if ~exist('traces', 'var')
        error('Missing traces argument.')
    end
    traces_attr = {'size', [2, NaN], 'nonnegative'};
    validateattributes(traces, {'numeric'}, traces_attr, '', 'traces');

    if ~exist('n_sectors', 'var')
        error('Missing n_sectors argument.')
    end
    n_sectors_attr = {'integer', 'positive', 'numel', 2};
    validateattributes(n_sectors, {'numeric'}, n_sectors_attr, '', 'n_sectors');

    % parse optional inputs
    parser = inputParser;
    posint_attr = {'scalar', 'positive', 'integer'};
    parser.addParameter('n_dct', 1, ...
        @(x) validateattributes(x, {'numeric'}, posint_attr, '', 'n_dct'));
    parser.addParameter('n_samples', 1, ...
        @(x) validateattributes(x, {'numeric'}, posint_attr, '', 'n_samples'));
    parser.addParameter('maxiter', 10000, ...
        @(x) validateattributes(x, {'numeric'}, posint_attr, '', 'maxiter'));
    parser.addParameter('winsize', 200, ...
        @(x) validateattributes(x, {'numeric'}, posint_attr, '', 'winsize'));
    pos_attr = {'scalar', 'positive'};
    parser.addParameter('adam_step', 1e-2, ...
        @(x) validateattributes(x, {'numeric'}, pos_attr, '', 'adam_step'));
    parser.addParameter('tolerance', 1e-4, ...
        @(x) validateattributes(x, {'numeric'}, pos_attr, '', 'tolerance'));
    verbose_class = {'logical', 'numeric'};
    verbose_attr = {'scalar', 'integer', 'nonnegative'};
    parser.addParameter('verbose', false, ...
        @(x) validateattributes(x, verbose_class, verbose_attr, '', 'verbose'));

    parser.parse(varargin{:});
    n_dct = parser.Results.n_dct;
    n_samples = parser.Results.n_samples;
    maxiter = parser.Results.maxiter;
    winsize = parser.Results.winsize;
    adam_step = parser.Results.adam_step;
    tolerance = parser.Results.tolerance;
    verbose = parser.Results.verbose;

    % rescale traces to correspond to prior assumptions
    scale = std(traces(:));
    traces = traces / scale;
    min_traces = min(traces(:));
    traces = traces - min_traces + 1;

    % create likelihood of the model and corresponding ELBO cost function
    dct_basis = make_dct_basis(n_dct, size(traces, 2));
    function [logp, g_params] = logpdf_fn(params, ~)
        [logp, g_params] = logpdf_n_grad(traces, n_sectors, dct_basis, params);
    end

    n_params = size(traces, 2) + n_dct * 2 + 2;
    prior_mean = zeros(1, n_params);
    prior_log_std = zeros(1, n_params);
    elbo_fn = make_elbo(@logpdf_fn, prior_mean, prior_log_std, n_samples);

    % fixed seed for reproducibility
    rng(12345);

    %  stochastic gradient descent
    init_params = [-1 * ones(1, n_params), -5 * ones(1, n_params)];
    adam = AdamOptimizer(init_params, 'step_size', adam_step);
    v_elbos = nan(1, maxiter);
    old_elbo_avg = -inf;

    for ii = 1:maxiter
        [v_elbo, g_elbo] = elbo_fn(adam.x, ii);
        v_elbos(ii) = v_elbo;
        adam.step(-g_elbo);

        % stop if smoothed ELBO don't improve enough
        if rem(ii, winsize) == 0
            new_elbo_avg = mean(v_elbos(ii-winsize+1:ii));
            if abs(old_elbo_avg - new_elbo_avg) < tolerance * abs(old_elbo_avg)
                break;
            end
            old_elbo_avg = new_elbo_avg;
        end

        % display extra information
        if verbose && rem(ii, 100) == 0
            [coeff, mu] = transform_params(adam.x(1:n_params)', n_dct);
            fprintf('iteration %d, elbo %f, coeff %f\n', ii, v_elbo, coeff);

            if verbose > 1
                plot_traces(traces, coeff, mu, v_elbos);
                pause(0.01);
            end
        end
    end

    % unpack results and clean traces
    var_params = adam.x';
    post_mean = var_params(1:n_params);
    [coeff, mu] = transform_params(post_mean, n_dct);
    cleaned_trace = (traces(1, :) - coeff * mu - 1 + min_traces) * scale;
end

function [coeff, mu, offset, sigma] = split_params(params, n_dct)
    % helper function to unpack parameters
    coeff = params(:, 1);
    mu = params(:, 2:end-2*n_dct-1);
    offset = params(:, end-2*n_dct:end-1);
    sigma = params(:, end);
end

function [coeff, mu, offset, sigma] = transform_params(params, n_dct)
    % helper function to unpack and transform parameters
    [coeff, mu, offset, sigma] = split_params(params, n_dct);
    coeff = normcdf(coeff);
    sigma = exp(sigma);
end

function [logp, g_x] = logpdf_n_grad(traces, n_sectors, dct_basis, x)
    % evaluate log-density of AST model and its gradient wrt. mu and sigma

    nt = size(traces, 2);
    y = reshape(traces, 1, 2, []);
    n = reshape(n_sectors, 1, 2);
    n_dct = size(dct_basis, 1);

    [coeff, mu, offset, sigma] = transform_params(x, n_dct);
    mu = reshape(mu, [], 1, nt);
    offset = reshape(offset, [], 2, n_dct);
    dct_basis = reshape(dct_basis, 1, 1, n_dct, []);

    % log-density and its gradient
    coeffs = cat(2, coeff, ones(size(coeff)));
    offsets = reshape(sum(offset .* dct_basis, 3), [], 2, nt);
    mus = coeffs .* mu + offsets;
    sigmas = sigma ./ sqrt(n);
    [logp, g_mus, g_sigmas] = ast_logpdf_n_grad(mus, sigmas, y);

    logp = sum(reshape(logp, size(x, 1), []), 2);

    g_coeffs = sum(g_mus .* mu, 3);
    g_coeff = g_coeffs(:, 1);
    g_mu = sum(g_mus .* coeffs, 2);
    g_offset = sum(reshape(g_mus, [], 2, 1, nt) .* dct_basis, 4);
    g_sigma = sum(sum(g_sigmas ./ sqrt(n), 3), 2);

    g_x = [g_coeff, g_mu(:, :), g_offset(:, :), g_sigma];
    g_x(:, 1) = g_x(:, 1) .* exp(0.5 * -x(:, 1).^2) ./ sqrt(2 .* pi);
    g_x(:, end) = g_x(:, end) .* exp(x(:, end));
end

function plot_traces(traces, coeff, mu, v_elbos)
    % utility function to display results of ongoing optimization

    subplot(3, 1, 1)
    cla();
    plot(v_elbos);
    title('ELBO')

    subplot(3, 1, 2)
    cla()
    hold('on');
    plot(traces(2, :));
    plot(traces(2, :) - mu);
    title('neighborhood')

    subplot(3, 1, 3)
    cla()
    hold('on');
    plot(traces(1, :));
    plot(traces(1, :) - coeff * mu);
    title('ROI')
end

function M = make_dct_basis(n_dct, n_frames)
    % create a incomplete DCT basis
    % TODO check proper normalization
    M = zeros(n_dct, n_frames);
    M(1, :) = 1;
    for ii = 2:n_dct
        xs = (2 .* (0:n_frames-1) + 1) .* (ii - 1);
        M(ii, :) = sqrt(2) .* cos(pi .* xs / (2 * n_frames));
    end
end