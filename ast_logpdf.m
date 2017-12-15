function logp = ast_logpdf(alpha, mu, nu1, nu2, sigma, y)
    % log-density of asymmetric Student-t distribution
    logp_left = ast_logpdf_left(alpha, mu, nu1, sigma, y);
    logp_right = ast_logpdf_right(alpha, mu, nu2, sigma, y);
    mask = y < mu;
    logp = zeros(size(logp_left));
    logp(mask) = logp_left(mask);
    logp(~mask) = logp_right(~mask);
end