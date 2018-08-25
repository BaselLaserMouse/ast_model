function [g_mu, g_sigma] = ast_logpdf_grad(mu, sigma, y)
    % gradient of log-density of asymmetric Student-t distribution
    mask = y < mu;

    g_mu_left = ast_logpdf_left_grad_mu(mu, sigma, y);
    g_mu_right = ast_logpdf_right_grad_mu(mu, sigma, y);
    g_mu = mask .* g_mu_left + ~mask .* g_mu_right;

    g_sigma_left = ast_logpdf_left_grad_sigma(mu, sigma, y);
    g_sigma_right = ast_logpdf_right_grad_sigma(mu, sigma, y);
    g_sigma = mask .* g_sigma_left + ~mask .* g_sigma_right;
end