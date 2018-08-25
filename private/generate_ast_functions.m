%% asymmetric Student-t distribution from Zhu and Galbraith (2009)

% log of the density function
syms y alpha nu1 nu2 mu sigma;

z = (y - mu) ./ (2 .* sigma);

K = @(nu) gamma(0.5 .* (nu + 1)) ./ (sqrt(pi .* nu) .* gamma(0.5 .* nu));

left_branch = ...
    -log(sigma) ...
    - 0.5 .* (nu1 + 1) .* log(1 + (z ./ (alpha .* K(nu1))).^2 ./ nu1);
left_branch = subs(left_branch, {alpha, nu1}, [0.5, 30]);

right_branch = ...
    -log(sigma) ...
    - 0.5 .* (nu2 + 1) .* log(1 + (z ./ ((1 - alpha) .* K(nu2))).^2 ./ nu2);
right_branch = subs(right_branch, {alpha, nu2}, [0.5, 1]);

% generate function files
matlabFunction(left_branch, 'File', 'ast_logpdf_left', 'Outputs', {'x'});
matlabFunction(right_branch, 'File', 'ast_logpdf_right', 'Outputs', {'x'});

% generate files for the gradient wrt. mu and sigma
matlabFunction(jacobian(left_branch, mu), ...
               'File', 'ast_logpdf_left_grad_mu', 'Outputs', {'x'});
matlabFunction(jacobian(right_branch, mu), ...
               'File', 'ast_logpdf_right_grad_mu', 'Outputs', {'x'});
matlabFunction(jacobian(left_branch, sigma), ...
               'File', 'ast_logpdf_left_grad_sigma', 'Outputs', {'x'});
matlabFunction(jacobian(right_branch, sigma), ...
               'File', 'ast_logpdf_right_grad_sigma', 'Outputs', {'x'});
