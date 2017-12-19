function [coeff, mu, offset, sigma] = fit_ast_model(traces, n_sectors, n_samples)
    % TODO documentation
    % TODO input checks

    if ~exist('n_samples', 'var')
        n_samples = 1;
    end

    % fix random seed to get reproducible gradients
    rng(12345);

    function [logp, g_x] = logpdf_fn(params, ~)
        [logp, g_x] = logpdf_n_grad(traces, n_sectors, params);
    end

    D = 4 + size(traces, 2);
    prior_mean = zeros(1, D);
    prior_log_std = ones(1, D);
    elbo_fn = make_elbo(@logpdf_fn, prior_mean, prior_log_std, n_samples);
    v_elbos = nan(1, 10000);

    function [val, g] = optim_fn(params, t)
        [val, g] = elbo_fn(params, t);
        v_elbos(t) = -val;
        if rem(t, 100) == 0
            plot_traces(traces, params, v_elbos, t);
        end
    end

    % initial parameters
    init_mean = -1 * ones(1, D);
    init_log_std = -5 * ones(1, D);
    init_params = cat(2, init_mean, init_log_std);

    %  stochastic gradient descent
    var_params = fmin_adam(@optim_fn, init_params', 1e-2);
    post_mean = var_params(1:D)';
    post_log_std = var_params(D:end)';
    [coeff, mu, offset, sigma] = transform_params(post_mean);

    % TODO apply correction
end

function [coeff, mu, offset, sigma] = split_params(params)
    coeff = params(:, 1);
    mu = params(:, 2:end-3);
    offset = params(:, end-2:end-1);
    sigma = params(:, end);
end

function [coeff, mu, offset, sigma] = transform_params(params)
    [coeff, mu, offset, sigma] = split_params(params);
    coeff = normcdf(coeff);
    mu = exp(mu);
    offset = exp(offset);
    sigma = exp(sigma);
end

function [logp, g_x] = logpdf_n_grad(traces, n_sectors, x)
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
    D = numel(params) / 2;
    [coeff, mu, offset] = transform_params(params(1:D)');
    fprintf('iteration %d, coeff %f\n', t, coeff);

    subplot(3, 1, 1)
    cla();
    plot(v_elbos);

    subplot(3, 1, 2)
    cla()
    hold('on');
    plot(traces(2, :));
    plot(traces(2, :) - mu - offset(2));

    subplot(3, 1, 3)
    cla()
    hold('on');
    plot(traces(1, :));
    plot(traces(1, :) - mu * coeff - offset(1));

    pause(1e-5)
end