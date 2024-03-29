%% benchmark on synthetic data, varying noise and contamination levels

% fix the seed for reproducibility
rng(1);

% benchmark parameters
nt = 5000;
n_noise_std = 5;
n_npil_ratio = 6;
n_replicates = 20;

noise_std = linspace(0, 1, n_noise_std);
npil_ratio = linspace(0, 1, n_npil_ratio);

% initialize result variables
mse_nothing = zeros(n_replicates, n_noise_std, n_npil_ratio);
mse_detrend = zeros(n_replicates, n_noise_std, n_npil_ratio);
mse_ast = zeros(n_replicates, n_noise_std, n_npil_ratio);
mse_best = zeros(n_replicates, n_noise_std, n_npil_ratio);

% loop over all configurations
tic;
for ii = 1:n_replicates
    for jj = 1:n_noise_std
        for kk = 1:n_npil_ratio
            disp([ii, jj, kk])

            [traces, cell_trace, cell_trend, cell_npil] = fake_data( ...
                nt, noise_std(jj), npil_ratio(kk), ii);

            cleaned_trace = fit_ast_model(traces, [1, 1], 'n_dct', 5);
            detrended_trace = traces(1, :) - cell_trend;
            best_trace = traces(1, :) - cell_npil - cell_trend;

            mse_nothing(ii, jj, kk) = mse(cell_trace - traces(1, :));
            mse_detrend(ii, jj, kk) = mse(cell_trace - detrended_trace);
            mse_ast(ii, jj, kk) = mse(cell_trace - cleaned_trace);
            mse_best(ii, jj, kk) = mse(cell_trace - best_trace);
        end
    end
end
toc

% save results
save('benchmark.mat', 'mse_nothing', 'mse_detrend', 'mse_ast', 'mse_best');

%% display results

% main figure
figure('Position', [344, 683, 1640, 406]);

fig_axes = cell(1, n_noise_std);
for ii = 1:n_noise_std
    fig_axes{ii} = subplot(1, n_noise_std, ii);
    hold('on');
    grid();
    h_nothing = plot_trace(npil_ratio, mse_nothing(:, ii, :));
    h_detrend = plot_trace(npil_ratio, mse_detrend(:, ii, :));
    h_ast = plot_trace(npil_ratio, mse_ast(:, ii, :));
    h_best = plot_trace(npil_ratio, mse_best(:, ii, :));
    xlabel('contamination ratio');
    ylabel('MSE');
    title(sprintf('noise std = %.3f', noise_std(ii)));
end
linkaxes(cat(1, fig_axes{:}), 'xy');
legend([h_nothing, h_detrend, h_ast, h_best], ...
       {'nothing', 'detrend', 'ASt', 'best'}, 'Location', 'southeast');

%% display typical data for each noise level

figure('Position', [50, 358, 2117, 889])

for ii = 1:n_noise_std
    [traces, cell_trace, cell_trend, cell_npil] = ...
        fake_data(nt, noise_std(ii), 0.5, 1);
    subplot(n_noise_std, 1, ii);
    hold('on');
    grid();
    plot(traces(1, :));
    plot(cell_trace, 'LineWidth', 2);
    ylabel(sprintf('noise std = %.3f', noise_std(ii)));
end

xlabel('frames');
legend('raw trace', 'true trace');

%% functions used in this script

function [traces, cell_trace, cell_trend, cell_npil] = fake_data(nt, noise_std, npil_ratio, seed)
    % create fake data with Poisson count statistics

    rng(seed);

    % cell soma data
    cell_trace = calcium_trace(0.01, nt) * 2 + 2;
    cell_trend = 1.1 - (1:nt).^2 ./ nt.^2;
    cell_trend = cell_trend - mean(cell_trend);

    % neighboring neuropil data
    npil_trace = calcium_trace(0.01, nt) + sin((1:nt) / 20) * 0.2;
    neighbor_trace = calcium_trace(0.01, nt) + 1;
    neighbor_trend = (1:nt).^2 ./ nt.^2;
    neighbor_trend = neighbor_trend - mean(neighbor_trend);

    % final noisy dataset
    cell_npil = npil_ratio * npil_trace;
    cell_noisy_trace = cell_trace + cell_trend + cell_npil;
    neighbor_noisy_trace = neighbor_trace + neighbor_trend + npil_trace;
    traces = [cell_noisy_trace; neighbor_noisy_trace] + randn(2, nt) * noise_std;
end

function ca_trace = calcium_trace(lambda, nt)
    template = -exp(-(1:30)) + exp(-0.2 * (1:30));
    template = template / max(template);
    spikes = poissrnd(lambda, 1, nt);
    ca_trace = conv(spikes, template, 'same');
end

function h_line = plot_trace(npil_ratio, mse_data)
    mean_mse = squeeze(mean(mse_data, 1))';
    std_mse = squeeze(std(mse_data, [], 1))';
    h_patch = fill([npil_ratio, fliplr(npil_ratio)], ...
                         [mean_mse + std_mse, fliplr(mean_mse - std_mse)], 'r');
    h_patch.FaceAlpha = 0.2;
    h_patch.LineStyle = 'none';
    h_line = plot(npil_ratio, mean_mse, '-o');
    h_patch.FaceColor = h_line.Color;
end