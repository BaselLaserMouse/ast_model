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

    function [v_elbo, g_elbo] = elbo_n_grad(params, t)
        rng(t);

        % sampling from approximate posterior
        post_mean = params(1:D)';
        post_log_std = params(D+1:end)';
        rvs = randn(n_samples, D);
        x = rvs .* exp(post_log_std) + post_mean;

        % unpack parameters samples
        x_exp = exp(x);
        coeff = x_exp(:, 1);
        mu = reshape(x_exp(:, 2:1+nt), [], 1, nt);
        offset = x_exp(:, 2+nt:3+nt);
        sigma = x_exp(:, end);

        disp(exp(post_mean(1)))
        cla();
        plot(squeeze(mean(mu, 1)));
        pause(0.1)

        coeffs = cat(2, coeff, ones(n_samples, 1));
        mus = coeffs .* mu + offset;
        sigmas = sigma ./ sqrt(n);

        % ELBO estimation
        logpdf = ast_logpdf(0.5, mus, 30, 1, sigmas, y);
        kl = kl_gauss(post_mean, post_log_std, prior_mean, prior_log_std);
        v_elbo = mean(sum(sum(logpdf, 3), 2), 1) - sum(kl);

        % gradient estimation
        [g_mus,g_sigmas] = ast_logpdf_grad(0.5, mus, 30, 1, sigmas, y);

        g_coeffs = sum(g_mus .* mu, 3);
        g_coeff = g_coeffs(:, 1);
        g_mu = sum(g_mus .* coeffs, 2);
        g_offset = sum(g_mus, 3);
        g_sigma = sum(sum(g_sigmas ./ sqrt(n), 3), 2);
        g_x = [g_coeff, g_mu(:, :), g_offset, g_sigma] .* x_exp;

        g_params = [mean(g_x, 1), mean(g_x .* rvs, 1) .* exp(post_log_std)];
        [g_post_mean, g_post_log_std] = ...
            kl_gauss_grad(post_mean, post_log_std, prior_mean, prior_log_std);
        g_kl = [g_post_mean, g_post_log_std];
        g_elbo = g_params - g_kl;

        % inverse signs to make it a minimization problem
        v_elbo = -v_elbo;
        g_elbo = -g_elbo;
    end

    % initial parameters
    init_mean = -1 * ones(1, D);
    init_log_std = -5 * ones(1, D);
    init_params = cat(2, init_mean, init_log_std);

    %  stochastic gradient descent
    var_params = fmin_adam(@elbo_n_grad, init_params');
%     var_params = init_params';
%     for ii = 1:10000
%         [v_elbo, g_elbo] = elbo_n_grad(var_params, ii);
%         fprintf('%d:  %f\n', ii, v_elbo);
%         var_params = var_params - g_elbo' * 1e-4;
% %         cla();
% %         plot(exp(var_params(2:1+nt)));
% %         pause(0.1)
%     end
    post_mean = var_params(1:D)';
    post_log_std = var_params(D:end)';

    coeff = exp(post_mean(1));
    mu = exp(post_mean(2:1+nt))';
    offset = exp(post_mean(2+nt:3+nt))';
    sigma = exp(post_mean(end));

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