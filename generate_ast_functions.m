%% asymmetric Student-t distribution from Zhu and Galbraith (2009)

% log of the density function
syms y alpha nu1 nu2 mu sigma;

z = (y - mu) ./ ( 2 .* sigma);

K = @(nu) gamma(0.5 .* (nu + 1)) ./ (sqrt(pi .* nu) .* gamma(0.5 .* nu));

left_branch = ...
    -log(sigma) ...
    - 0.5 .* (nu1 + 1) .* log(1 + (z ./ (alpha .* K(nu1))).^2 ./ nu1);

right_branch = ...
    -log(sigma) ...
    - 0.5 .* (nu2 + 1) .* log(1 + (z ./ ((1 - alpha) .* K(nu2))).^2 ./ nu2);

% generate function file
matlabFunction(left_branch, 'File', 'ast_logpdf_left', 'Outputs', {'x'})
matlabFunction(right_branch, 'File', 'ast_logpdf_right', 'Outputs', {'x'})

% generate file for the gradient wrt. mu and sigma
matlabFunction(jacobian(left_branch, [mu, sigma]), ...
               'File', 'ast_logpdf_left_grad', 'Outputs', {'x'});
matlabFunction(jacobian(right_branch, [mu, sigma]), ...
               'File', 'ast_logpdf_right_grad', 'Outputs', {'x'});