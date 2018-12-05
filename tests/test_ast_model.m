%% create random data

rng(1);

nt = 1000;
npil_sig = abs(randn(1, nt)) * 30;
cell_sig = abs(randn(1, nt)) * 10;

traces = [cell_sig + npil_sig * 0.7 + 5 + randn(1, nt);
          npil_sig + 7 + randn(1, nt)];

%% create a more involved test data set

rng(1);

nt = 1000;
noise_std = 0.1;

template = -exp(-(1:30)) + exp(-0.2 * (1:30));

cell_spikes = poissrnd(0.01, 1, nt);
cell_trace = conv(cell_spikes, template, 'same');
cell_trend = 1.1 - (1:nt).^2 ./ nt.^2 + 5;
cell_noise = randn(1, nt) * noise_std;

npil_spikes = poissrnd(0.005, 1, nt);
npil_trace = conv(npil_spikes, template, 'same');
npil_trend = (1:nt).^2 ./ nt.^2 + 3;
npil_noise = randn(1, nt) * noise_std;

traces = [cell_trace + cell_trend + cell_noise + 0.9 * npil_trace;
          npil_trace + npil_trend + npil_noise];

%% try to decontaminate signal

cleaned_trace = fit_ast_model(traces, [1, 1], 'verbose', 2, 'n_dct', 5);