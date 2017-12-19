function elbo_fn = make_elbo(logpdf_n_grad, prior_mean, prior_log_std, n_samples)

    function [val, g] = elbo_optim_fn(params, t)
        % compute ELBO value and gradient
        [v_elbo, g_elbo] = elbo_n_grad( ...
            logpdf_n_grad, prior_mean, prior_log_std, n_samples, params, t);
        % inverse signs to make it a minimization problem
        val = -v_elbo;
        g = -g_elbo;
    end

    elbo_fn = @elbo_optim_fn;
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

function [v_elbo, g_elbo] = elbo_n_grad(logpdf_n_grad, prior_mean, prior_log_std, n_samples, params, t)
    % sampling from approximate posterior
    D = numel(params) / 2;
    post_mean = params(1:D)';
    post_log_std = params(D+1:end)';
    rvs = randn(n_samples, D);
    x = rvs .* exp(post_log_std) + post_mean;

    % ELBO estimation
    [logp, g_x] = logpdf_n_grad(x, t);
    kl = kl_gauss(post_mean, post_log_std, prior_mean, prior_log_std);
    v_elbo = mean(logp) - sum(kl);

    % gradient estimation
    g_params = [mean(g_x, 1), mean(g_x .* rvs, 1) .* exp(post_log_std)];
    [g_post_mean, g_post_log_std] = ...
        kl_gauss_grad(post_mean, post_log_std, prior_mean, prior_log_std);
    g_kl = [g_post_mean, g_post_log_std];
    g_elbo = g_params - g_kl;
end