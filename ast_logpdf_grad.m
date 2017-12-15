function logp_grad = ast_logpdf_grad(alpha, mu, nu1, nu2, sigma, y)
    % gradient of log-density of asymmetric Student-t distribution
    logp_grad_left = ast_logpdf_left_grad(alpha, mu, nu1, sigma, y);
    logp_grad_right = ast_logpdf_right_grad(alpha, mu, nu2, sigma, y);
    mask = y < mu;
    logp_grad = zeros(size(logp_grad_left));
    logp_grad(mask) = logp_grad_left(mask);
    logp_grad(~mask) = logp_grad_right(~mask);
end