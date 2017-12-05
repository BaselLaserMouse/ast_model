function [coeff, mu, offset, sigma] = fit_ast_model(y, n)
    % TODO documentation
    % TODO input checks

    nt = size(y, 2);
    D = 4 + nt;

    y = reshape(y, 1, 2, []);
    n = reshape(n, 1, 2, 1);

    function [v_elbo, g_elbo] = elbo_n_grad(params)
        % sampling from approximate posterior
        post_mean = params(1:D);
        post_log_std = params(D+1:end);
        rvs = randn(n_samples, D);
        x = rvs .* exp(post_log_std) + post_mean;

        % unpack parameters samples
        coeff = reshape(exp(x(:, 1)), [], 1, 1);
        mu = reshape(exp(x(:, 2:1+nt)), [], 1, nt);
        offset = reshape(exp(:, 2+nt:3+nt), [], 2, 1);
        sigma = reshape(exp(:, end), [], 1, 1);

        coeffs = cat(2, coeff, ones(1, n_samples, 1));
        mus = coeffs .* mu + offset;
        sigmas = sigma ./ sqrt(n);

        % ELBO estimation
        logpdf = ast_logpdf(0.5, mus, 30, 1, sigmas, y);
        v_elbo = mean(sum(sum(logpdf, 3), 2), 1);
            kl_gauss(prior_mean, prior_log_std, post_mean, post_log_std);

        % gradient estimation
        g_logpdf = mean(ast_logpdf_grad(0.5, mus, 30, 1, sigmas, y), 1);
        g_mus = g_logpdf(1, :, :);
        g_sigmas = g_logpdf(1, :, :);
        g_coeff = sum(g_mus * mu, 3);
        g_mu = sum(g_mus * coeffs, 2);
        g_offset = sum(g_mus, 3);
        g_sigma = sum(sum(g_sigmas ./ sqrt(n), 3), 2);
        g_x = [g_coeff(:); g_mu(:); g_offset(:); g_sigma(:)] .* exp(x);
        g_params = [g_x; g_x .* rvs .* exp(post_log_std)];

        g_elbo = g_params + ...
            kl_gauss_grad(prior_mean, prior_log_std, post_mean, post_log_std);
    end
end

function x = kl_gauss(mean1, log_std1, mean2, log_std2)
    % KL divergence between to normal distribution
end

function x = kl_gauss_grad(mean1, log_std1, mean2, log_std2)
    % gradient of KL divergence between to normal distribution, wrt. 2nd one
end