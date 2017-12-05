function x = ast_logpdf_grad(alpha, mu, nu1, nu2, sigma, y)
    % gradient of log-density of asymmetric Student-t distribution
    if isvector(y)
        x = zeros([2, numel(y)]);
    else
        x = zeros([2, size(y)]);
    end

    mask_left = y < mu;
    y_left = reshape(y(mask_left), [], 1);
    x(:, mask_left) = ast_logpdf_left_grad(alpha, mu, nu1, sigma, y_left)';
    y_right = reshape(y(~mask_left), [], 1);
    x(:, ~mask_left) = ast_logpdf_right_grad(alpha, mu, nu2, sigma, y_right)';
end