function [coeff, mu, offset, sigma] = fit_ast_model(traces, n_sectors, n_samples)
    % TODO documentation
    % TODO input checks

    if ~exist('n_samples', 'var')
        n_samples = 1;
    end

    nt = size(traces, 2);
    D = 4 + nt;

    prior_mean = zeros(1, D);
    prior_log_std = ones(1, D);

    function [coeff, mu, offset, sigma] = split_params(params)
        coeff = params(:, 1);
        mu = params(:, 2:1+nt);
        offset = params(:, 2+nt:3+nt);
        sigma = params(:, end);
    end

    function [coeff, mu, offset, sigma] = transform_params(params)
        [coeff, mu, offset, sigma] = split_params(params);
        coeff = normcdf(coeff);
        mu = exp(mu);
        offset = exp(offset);
        sigma = exp(sigma);
    end

    function [logp, g_x] = logpdf_n_grad(x)
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

    v_elbos = nan(1, 10000);

    function callback(params, t)
        if rem(t, 100) ~= 0
            return;
        end

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

        pause(0.01)
    end

    function [v_elbo, g_elbo] = elbo_n_grad(params, t)
        % fix random seed to get reproducible gradients
        rng(t);

        % sampling from approximate posterior
        post_mean = params(1:D)';
        post_log_std = params(D+1:end)';
        rvs = randn(n_samples, D);
        x = rvs .* exp(post_log_std) + post_mean;

        % ELBO estimation
        [logp, g_x] = logpdf_n_grad(x);
        kl = kl_gauss(post_mean, post_log_std, prior_mean, prior_log_std);
        v_elbo = mean(logp) - sum(kl);

        % gradient estimation
        g_params = [mean(g_x, 1), mean(g_x .* rvs, 1) .* exp(post_log_std)];
        [g_post_mean, g_post_log_std] = ...
            kl_gauss_grad(post_mean, post_log_std, prior_mean, prior_log_std);
        g_kl = [g_post_mean, g_post_log_std];
        g_elbo = g_params - g_kl;

        % inverse signs to make it a minimization problem
        v_elbo = -v_elbo;
        g_elbo = -g_elbo;

        % display some feedback
        v_elbos(t) = -v_elbo;
        callback(params, t);
    end

    % initial parameters (coeff, mu, offset, sigma)
%     init_mu = log(squeeze(y(1, 2, :)) - min(y(1, 2, :)) + 1e-6)';
%     init_mean = [log(0.7), init_mu, log(min(y, [], 3)), log(1)];
    init_mean = -1 * ones(1, D);
    init_log_std = -5 * ones(1, D);
    init_params = cat(2, init_mean, init_log_std);

    %  stochastic gradient descent
    elbo_n_grad(init_params', 1);
    var_params = fmin_adam(@elbo_n_grad, init_params', 1e-2);
    post_mean = var_params(1:D)';
    post_log_std = var_params(D:end)';
    [coeff, mu, offset, sigma] = transform_params(post_mean);

    % TODO apply correction
end

function x = kl_gauss(mean1, log_std1, mean2, log_std2)
    % KL divergence between to normal distribution
    x = 0.5 * ((mean2 - mean1).^2 + exp(2 * log_std1)) ./ exp(2 * log_std2) ...
        + log_std2 - log_std1 - 0.5;
end

function [g_mean1, g_log_std1] = kl_gauss_grad(mean1, log_std1, mean2, log_std2)
    % gradient of KL divergence between to normal distribution, wrt. 2nd one
    g_mean1 = exp(-2 .* log_std2) .* (mean1 - mean2);
    g_log_std1 = exp(2 .* log_std1) .* exp(-2 .* log_std2) - 1;
end