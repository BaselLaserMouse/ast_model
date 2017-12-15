function [coeff, mu, offset, sigma] = fit_ast_model(y, n)
    % TODO documentation
    % TODO input checks

    n_samples = 10;
    nt = size(y, 2);
    D = 4 + nt;

    y = reshape(y, 1, 2, []);
    n = reshape(n, 1, 2);

    prior_mean = zeros(1, D);
    prior_log_std = ones(1, D);

    function [v_elbo, g_elbo] = elbo_n_grad(params)
        % sampling from approximate posterior
        post_mean = params(1, 1:D);
        post_log_std = params(1, D+1:end);
        rvs = randn(n_samples, D);
        x = rvs .* exp(post_log_std) + post_mean;

        % unpack parameters samples
        x_exp = exp(x);
        coeff = x_exp(:, 1);
        mu = reshape(x_exp(:, 2:1+nt), [], 1, nt);
        offset = x_exp(:, 2+nt:3+nt);
        sigma = x_exp(:, end);

        coeffs = cat(2, coeff, ones(n_samples, 1));
        mus = coeffs .* mu + offset;
        sigmas = sigma ./ sqrt(n);

        % ELBO estimation
        logpdf = ast_logpdf(0.5, mus, 30, 1, sigmas, y);
        kl = kl_gauss(post_mean, post_log_std, prior_mean, prior_log_std);
        v_elbo = mean(sum(sum(logpdf, 3), 2), 1) - sum(kl);

        % gradient estimation
        g_logpdf = ast_logpdf_grad(0.5, mus, 30, 1, sigmas, y);

        g_mus = g_logpdf(:, 1:2, :);
        g_sigmas = g_logpdf(:, 3:4, :);
        g_coeffs = sum(g_mus .* mu, 3);

        g_coeff = g_coeffs(:, 1);
        g_mu = sum(g_mus .* coeffs, 2);
        g_offset = sum(g_mus, 3);
        g_sigma = sum(sum(g_sigmas ./ sqrt(n), 3), 2);
        g_x = [g_coeff, g_mu(:, :), g_offset, g_sigma] .* x_exp;

        g_params = [sum(g_x, 1), sum(g_x .* rvs, 1) .* exp(post_log_std)];
        g_kl = kl_gauss_grad(post_mean, post_log_std, prior_mean, prior_log_std);
        g_elbo = g_params - g_kl;
    end

    % TODO initial parameters
    init_mean = -1 * ones(1, D);
    init_log_std = -5 * ones(1, D);
    init_params = cat(2, init_mean, init_log_std);

    % TODO stochastic gradient descent

    % TODO apply correction
end

function x = kl_gauss(mean1, log_std1, mean2, log_std2)
    % KL divergence between to normal distribution
    x = 0.5 * ((mean2 - mean1).^2 + exp(2 * log_std1)) ./ exp(2 * log_std2) ...
        + log_std2 - log_std1 - 0.5;
end

function x = kl_gauss_grad(mean1, log_std1, mean2, log_std2)
    % gradient of KL divergence between to normal distribution, wrt. 2nd one
end