function logp = ast_logpdf(mu, sigma, y)
    % log-density of asymmetric Student-t distribution
    logp_left = ast_logpdf_left(mu, sigma, y);
    logp_right = ast_logpdf_right(mu, sigma, y);
    mask = y < mu;
    logp = mask .* logp_left + ~mask .* logp_right;
end