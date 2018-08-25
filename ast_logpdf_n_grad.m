function [logp, g_mu, g_sigma] = ast_logpdf_n_grad(mu, sigma, y)
    % log-density of asymmetric Student-t distribution and its gradient
    mask = y < mu;

    [logp_left, g_mu_left, g_sigma_left] = ast_logpdf_n_grad_left(mu, sigma, y);
    [logp_right, g_mu_right, g_sigma_right] = ast_logpdf_n_grad_right(mu, sigma, y);

    logp = mask .* logp_left + ~mask .* logp_right;
    g_mu = mask .* g_mu_left + ~mask .* g_mu_right;
    g_sigma = mask .* g_sigma_left + ~mask .* g_sigma_right;
end