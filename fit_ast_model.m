function [cleaned_trace, var_params, v_elbos] = fit_ast_model(traces, n_sectors, varargin)
    % TODO documentation

    % TODO meaningful priors
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
    parser.addParameter('n_samples', 1, ...
        @(x) validateattributes(x, {'numeric'}, posint_attr, '', 'n_samples'));
    parser.addParameter('maxiter', 10000, ...
        @(x) validateattributes(x, {'numeric'}, posint_attr, '', 'maxiter'));
    verbose_class = {'logical', 'numeric'};
    verbose_attr = {'scalar', 'integer', 'nonnegative'};
    parser.addParameter('verbose', false, ...
        @(x) validateattributes(x, verbose_class, verbose_attr, '', 'verbose'));

    parser.parse(varargin{:});
    n_samples = parser.Results.n_samples;
    maxiter = parser.Results.maxiter;
    verbose = parser.Results.verbose;

    % rescale traces to correspond to prior assumptions
    scale = std(traces(:));
    traces = traces / scale;
    min_traces = min(traces(:));
    traces = traces - min_traces + 1;

    % create likelihood of the model and corresponding ELBO cost function
    function [logp, g_params] = logpdf_fn(params, ~)
        [logp, g_params] = logpdf_n_grad(traces, n_sectors, params);
    end

    n_params = size(traces, 2) + 4;
    prior_mean = zeros(1, n_params);
    prior_log_std = zeros(1, n_params);
    elbo_fn = make_elbo(@logpdf_fn, prior_mean, prior_log_std, n_samples);

    % fixed seed for reproducibility
    rng(12345);

    %  stochastic gradient descent
    init_params = [-1 * ones(1, n_params), -5 * ones(1, n_params)];
    adam = AdamOptimizer(init_params, 'step_size', 1e-2);
    v_elbos = nan(1, maxiter);

    for ii = 1:maxiter
        [v_elbo, g_elbo] = elbo_fn(adam.x, ii);
        v_elbos(ii) = v_elbo;
        adam.step(-g_elbo);

        % display extra information
        if verbose && rem(ii, 100) == 0
            [coeff, mu] = transform_params(adam.x(1:n_params)');
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
    [coeff, mu] = transform_params(post_mean);
    cleaned_trace = (traces(1, :) - coeff * mu - 1 + min_traces) * scale;
end

function [coeff, mu, offset, sigma] = split_params(params)
    % helper function to unpack parameters
    coeff = params(:, 1);
    mu = params(:, 2:end-3);
    offset = params(:, end-2:end-1);
    sigma = params(:, end);
end

function [coeff, mu, offset, sigma] = transform_params(params)
    % helper function to unpack and transform parameters
    [coeff, mu, offset, sigma] = split_params(params);
    coeff = normcdf(coeff);
    sigma = exp(sigma);
end

function [logp, g_x] = logpdf_n_grad(traces, n_sectors, x)
    % evaluate log-density of AST model and its gradient wrt. mu and sigma

    nt = size(traces, 2);
    y = reshape(traces, 1, 2, []);
    n = reshape(n_sectors, 1, 2);

    [coeff, mu, offset, sigma] = transform_params(x);
    mu = reshape(mu, [], 1, nt);

    % log-density
    coeffs = cat(2, coeff, ones(size(coeff)));
    mus = coeffs .* mu + offset;
    sigmas = sigma ./ sqrt(n);
    logp = ast_logpdf(0.5, mus, 30, 1, sigmas, y);
    logp = sum(reshape(logp, size(x, 1), []));

    % gradient of log-density
    [g_mus, g_sigmas] = ast_logpdf_grad(0.5, mus, 30, 1, sigmas, y);

    g_coeffs = sum(g_mus .* mu, 3);
    g_coeff = g_coeffs(:, 1);
    g_mu = sum(g_mus .* coeffs, 2);
    g_offset = sum(g_mus, 3);
    g_sigma = sum(sum(g_sigmas ./ sqrt(n), 3), 2);

    g_x = [g_coeff, g_mu(:, :), g_offset, g_sigma];
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