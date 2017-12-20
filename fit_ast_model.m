function [coeff, mu, offset, sigma] = fit_ast_model(traces, n_sectors, n_samples)
    % TODO documentation
    % TODO input checks

    maxiter = 10000;

    if ~exist('n_samples', 'var')
        n_samples = 1;
    end

    function [logp, g_params] = logpdf_fn(params, ~)
        [logp, g_params] = logpdf_n_grad(traces, n_sectors, params);
    end

    n_params = size(traces, 2) + 4;
    prior_mean = zeros(1, n_params);
    prior_log_std = ones(1, n_params);
    elbo_fn = make_elbo(@logpdf_fn, prior_mean, prior_log_std, n_samples);

    % fixed seed for reproducibility
    rng(12345);

    %  stochastic gradient descent
    init_params = [-1 * ones(1, n_params), -5 * ones(1, n_params)];
    adam = AdamOptimizer(init_params, 'step_size', 1e-2);
    v_elbos = nan(1, maxiter);

    for ii = 1:maxiter
        [v_elbo, g_elbo] = elbo_fn(adam.x, ii);
        adam.step(-g_elbo);

        v_elbos(ii) = v_elbo;
        if rem(ii, 100) == 0
            plot_traces(traces, adam.x, v_elbos, ii);
            pause(0.01);
        end
    end

    % unpack results
    post_mean = adam.x(1:n_params)';
    post_log_std = adam.x(n_params:end)';
    [coeff, mu, offset, sigma] = transform_params(post_mean);

    % TODO apply correction
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
    mu = exp(mu);
    offset = exp(offset);
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
    g_x(:, 2:end) = g_x(:, 2:end) .* exp(x(:, 2:end));
end

function plot_traces(traces, params, v_elbos, t)
    % utiliy function to display results of ongoing optimization

    D = numel(params) / 2;
    [coeff, mu] = transform_params(params(1:D)');
    fprintf('iteration %d, coeff %f\n', t, coeff);

    subplot(3, 1, 1)
    cla();
    plot(v_elbos);

    subplot(3, 1, 2)
    cla()
    hold('on');
    plot(traces(2, :));
    plot(traces(2, :) - mu);

    subplot(3, 1, 3)
    cla()
    hold('on');
    plot(traces(1, :));
    plot(traces(1, :) - mu * coeff);
end