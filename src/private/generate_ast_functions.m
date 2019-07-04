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
matlabFunction(left_branch, jacobian(left_branch, mu), jacobian(left_branch, sigma), ...
    'File', 'ast_logpdf_n_grad_left', 'Outputs', {'logp', 'g_mu', 'g_sigma'});
matlabFunction(right_branch, jacobian(right_branch, mu), jacobian(right_branch, sigma), ...
    'File', 'ast_logpdf_n_grad_right', 'Outputs', {'logp', 'g_mu', 'g_sigma'});