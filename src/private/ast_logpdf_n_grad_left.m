function [logp,g_mu,g_sigma] = ast_logpdf_n_grad_left(mu,sigma,y)
%AST_LOGPDF_N_GRAD_LEFT
%    [LOGP,G_MU,G_SIGMA] = AST_LOGPDF_N_GRAD_LEFT(MU,SIGMA,Y)

%    This function was generated by the Symbolic Math Toolbox version 8.0.
%    25-Aug-2018 12:18:11

t2 = mu-y;
t3 = 1.0./sigma.^2;
t4 = t2.^2;
t5 = t3.*t4.*2.129587634366285e-1;
t6 = t5+1.0;
logp = -log(sigma)-log(t6).*(3.1e1./2.0);
if nargout > 1
    t7 = 1.0./t6;
    g_mu = t3.*t7.*(mu.*2.0-y.*2.0).*(-3.300860833267742);
end
if nargout > 2
    g_sigma = -1.0./sigma+1.0./sigma.^3.*t4.*t7.*6.601721666535483;
end
