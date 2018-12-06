%% benchmark on synthetic data, varying noise and contamination levels

% fix the seed for reproducibility
rng(1);

nt = 5000;
n_noise_std = 3;
n_npil_ratio = 4;
n_replicates = 2;

noise_std = logspace(-3, 0, n_noise_std);
npil_ratio = linspace(0, 1, n_npil_ratio);

results_mse = zeros(n_replicates, n_noise_std, n_npil_ratio);
for ii = 1:n_replicates
    for jj = 1:n_noise_std
        for kk = 1:n_npil_ratio
            disp([ii, jj, kk])
            traces = fake_data(nt, noise_std(jj), npil_ratio(kk), ii);
            cleaned_trace = fit_ast_model(traces, [1, 1], 'n_dct', 5);
            results_mse(ii, jj, kk) = mean((cell_trace - cleaned_trace).^2);
        end
    end
end

%% functions used in this script

function traces = fake_data(nt, noise_std, npil_ratio, seed)
    % create fake data with Poisson count statistics

    rng(seed);
    
    % calcium spike template
    template = -exp(-(1:30)) + exp(-0.2 * (1:30));

    % cell soma data
    cell_spikes = poissrnd(0.01, 1, nt);
    cell_trace = conv(cell_spikes, template, 'same') + 4;
    cell_trend = 1.1 - (1:nt).^2 ./ nt.^2;
    cell_trend = cell_trend - mean(cell_trend);

    % neighboring neuropil data
    npil_spikes = poissrnd(0.01, 1, nt);
    npil_trace = conv(npil_spikes, template, 'same') + sin((1:nt) / 20) * 0.2;
    npil_trend = (1:nt).^2 ./ nt.^2;
    npil_trend = npil_trend - mean(npil_trend);

    % final noisy dataset
    cell_noisy_trace = cell_trace + cell_trend + npil_ratio * npil_trace;
    npil_noisy_trace = npil_trace + npil_trend + 5;
    traces = [cell_noisy_trace; npil_noisy_trace] + randn(2, nt) * noise_std;
end