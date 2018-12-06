%% benchmark on synthetic data, varying noise and contamination levels

% fix the seed for reproducibility
rng(1);

nt = 5000;
n_noise_std = 5;
n_npil_ratio = 8;
n_replicates = 1;

noise_std = linspace(0, 1, n_noise_std);
npil_ratio = linspace(0, 1, n_npil_ratio);

mse_nothing = zeros(n_replicates, n_noise_std, n_npil_ratio);
mse_detrend = zeros(n_replicates, n_noise_std, n_npil_ratio);
mse_ast = zeros(n_replicates, n_noise_std, n_npil_ratio);
mse_best = zeros(n_replicates, n_noise_std, n_npil_ratio);

for ii = 1:n_replicates
    tic;
    for jj = 1:n_noise_std
        for kk = 1:n_npil_ratio
            disp([ii, jj, kk])

            [traces, cell_trace, cell_trend, cell_npil] = fake_data( ...
                nt, noise_std(jj), npil_ratio(kk), ii);
            cleaned_trace = fit_ast_model(traces, [1, 1], 'n_dct', 5);

            mse_nothing(ii, jj, kk) = mean((cell_trace - traces(1, :)).^2);
            detrended_trace = traces(1, :) - cell_trend;
            mse_detrend(ii, jj, kk) = mean((cell_trace - detrended_trace).^2);
            mse_ast(ii, jj, kk) = mean((cell_trace - cleaned_trace).^2);
            best_trace = traces(1, :) - cell_npil - cell_trend;
            mse_best(ii, jj, kk) = mean((cell_trace - best_trace).^2);
        end
    end
    toc
end

mean_mse_nothing = squeeze(mean(mse_nothing, 1));
mean_mse_detrend = squeeze(mean(mse_detrend, 1));
mean_mse_ast = squeeze(mean(mse_ast, 1));
mean_mse_best = squeeze(mean(mse_best, 1));

% TODO save resuls

%% display results

figure('Position',[344 683 1640 406]);

for ii = 1:n_noise_std
    subplot(1, n_noise_std, ii);
    hold('on');
    grid();
    plot(npil_ratio, mean_mse_nothing(ii, :), '-o');
    plot(npil_ratio, mean_mse_detrend(ii, :), '-o');
    plot(npil_ratio, mean_mse_ast(ii, :), '-o');
    plot(npil_ratio, mean_mse_best(ii, :), '-o');
    xlabel('contamination ratio');
    ylabel('MSE');
    title(sprintf('noise std = %.3f', noise_std(ii)));
end
legend('nothing', 'detrend', 'ASt', 'best');

%% functions used in this script

function [traces, cell_trace, cell_trend, cell_npil] = fake_data(nt, noise_std, npil_ratio, seed)
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
    cell_npil = npil_ratio * npil_trace;
    cell_noisy_trace = cell_trace + cell_trend + cell_npil;
    npil_noisy_trace = npil_trace + npil_trend + 5;
    traces = [cell_noisy_trace; npil_noisy_trace] + randn(2, nt) * noise_std;
end