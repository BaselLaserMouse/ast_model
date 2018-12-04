%% create random data

rng(1);

nt = 1000;
npil_sig = abs(randn(1, nt)) * 30;
cell_sig = abs(randn(1, nt)) * 10;

traces = [cell_sig + npil_sig * 0.7 + 5 + randn(1, nt);
          npil_sig + 7 + randn(1, nt)];

%% try to decontaminate signal

cleaned_trace = fit_ast_model(traces, [1, 1], 'verbose', 2, 'n_dct', 5);