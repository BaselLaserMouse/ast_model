function logp = ast_logpdf(alpha, mu, nu1, nu2, sigma, y)
    % log-density of asymmetric Student-t distribution
    logp = zeros(size(y));
    mask_left = y < mu;
    logp(mask_left) = ast_logpdf_left(alpha, mu, nu1, sigma, y(mask_left));
    logp(~mask_left) = ast_logpdf_right(alpha, mu, nu2, sigma, y(~mask_left));
end