%% multipath simulation
% multipath channel effects
% cyclic prefix testing

clear; close all; clc;

% load results
load('exercise_4_1_1_results.mat');

fprintf('=== Exercise 4.1.3: Multipath Channel Simulation ===\n');

%% multipath channel model
function output = multipath_channel(input, delays, gains)
    % multipath delays
    
    output = zeros(size(input));
    
    for i = 1:length(delays)
        if delays(i) == 0
            % direct path
            output = output + gains(i) * input;
        else
            % delayed path
            delayed_input = [zeros(delays(i), 1); input(1:end-delays(i))];
            output = output + gains(i) * delayed_input;
        end
    end
end

%% channel equalizer
function eq_symbols = simple_equalizer(rx_symbols, pilot_indices, pilot_values, active_indices)
    % zero forcing
    
    eq_symbols = rx_symbols;
    
    % extract pilots
    rx_pilots = rx_symbols(pilot_indices);
    
    % channel response
    channel_response_pilots = rx_pilots ./ pilot_values;
    
    % interpolate carriers
    channel_response_full = interp1(pilot_indices, channel_response_pilots, ...
                                   active_indices, 'linear', 'extrap');
    
    % zero forcing
    eq_symbols = rx_symbols ./ channel_response_full;
end

%% test parameters
T_guard_samples = N_guard;  % guard samples
max_delay_samples = round(T_guard_samples * 0.8);  % guard limit
delay_step = 5;  % delay steps

% test signal
N_test_symbols = 20;
test_signal = ofdm_signal(:, 1:min(N_test_symbols, size(ofdm_signal, 2)));
if size(test_signal, 2) < N_test_symbols
    % repeat signal
    test_signal = repmat(test_signal, 1, ceil(N_test_symbols / size(test_signal, 2)));
    test_signal = test_signal(:, 1:N_test_symbols);
end

% flatten signal
tx_signal = test_signal(:);

fprintf('Guard time: %d samples (%.2f μs)\n', T_guard_samples, T_guard_samples / fs * 1e6);
fprintf('Testing delays up to: %d samples (%.2f μs)\n', max_delay_samples, max_delay_samples / fs * 1e6);

%% multipath scenarios

% scenario 1
fprintf('\n--- Scenario 1: Single reflection (delay < guard time) ---\n');
delay1 = round(T_guard_samples * 0.5);  % guard percentage
gain1 = 0.7 * exp(1j * pi/4);  % amplitude phase

channel1_output = multipath_channel(tx_signal, [0, delay1], [1, gain1]);
fprintf('Delay: %d samples (%.2f μs)\n', delay1, delay1 / fs * 1e6);
fprintf('Relative gain: %.2f, Phase: %.1f°\n', abs(gain1), angle(gain1)*180/pi);

% Scenario 2: reflection exceeding guard  
fprintf('\n--- Scenario 2: Single reflection (delay > guard time) ---\n');
delay2 = round(T_guard_samples * 1.5);  % guard percentage
gain2 = 0.4 * exp(1j * pi/3);  % amplitude phase

channel2_output = multipath_channel(tx_signal, [0, delay2], [1, gain2]);
fprintf('Delay: %d samples (%.2f μs)\n', delay2, delay2 / fs * 1e6);
fprintf('Relative gain: %.2f, Phase: %.1f°\n', abs(gain2), angle(gain2)*180/pi);

% Scenario 3: multiple reflections
fprintf('\n--- Scenario 3: Multiple reflections ---\n');
delays3 = [0, round(T_guard_samples * 0.3), round(T_guard_samples * 0.6)];
gains3 = [1, 0.5 * exp(1j * pi/6), 0.3 * exp(1j * pi/2)];

channel3_output = multipath_channel(tx_signal, delays3, gains3);
fprintf('Path 1 - Delay: %d samples, Gain: %.2f, Phase: %.1f°\n', ...
        delays3(1), abs(gains3(1)), angle(gains3(1))*180/pi);
fprintf('Path 2 - Delay: %d samples, Gain: %.2f, Phase: %.1f°\n', ...
        delays3(2), abs(gains3(2)), angle(gains3(2))*180/pi);
fprintf('Path 3 - Delay: %d samples, Gain: %.2f, Phase: %.1f°\n', ...
        delays3(3), abs(gains3(3)), angle(gains3(3))*180/pi);

%% demodulation analysis
scenarios = {channel1_output, channel2_output, channel3_output};
scenario_names = {'Within Guard Time', 'Exceeding Guard Time', 'Multiple Paths'};
colors = {'b', 'r', 'g'};

% original reference
original_symbols = zeros(N_active, N_test_symbols);
for sym_idx = 1:N_test_symbols
    freq_domain = fft(test_signal(N_guard+1:end, sym_idx), N_fft);
    active_indices = (-N_active/2:N_active/2-1) + N_fft/2 + 1;
    original_symbols(:, sym_idx) = freq_domain(active_indices);
end

figure(1);
constellation_plots = subplot(2,3,1);
scatter(real(original_symbols(:)), imag(original_symbols(:)), 20, 'ko', 'filled');
title('Original Signal');
xlabel('In-phase'); ylabel('Quadrature');
grid on; axis equal; axis([-2 2 -2 2]);

evm_results = zeros(1, length(scenarios));
ber_results = zeros(1, length(scenarios));

for scenario_idx = 1:length(scenarios)
    rx_signal = scenarios{scenario_idx};
    
    % Demodulate received symbols
    rx_symbols = zeros(N_active, N_test_symbols);
    
    for sym_idx = 1:N_test_symbols
        % extract symbol
        start_idx = (sym_idx-1) * (N_fft + N_guard) + 1;
        end_idx = start_idx + N_fft + N_guard - 1;
        
        % boundary cases
        if end_idx > length(rx_signal)
            break;
        end
        
        rx_symbol = rx_signal(start_idx:end_idx);
        
        % remove prefix
        rx_useful = rx_symbol(N_guard+1:end);
        freq_rx = fft(rx_useful, N_fft);
        active_indices = (-N_active/2:N_active/2-1) + N_fft/2 + 1;
        rx_symbols(:, sym_idx) = freq_rx(active_indices);
    end
    
    % channel equalization
    if scenario_idx > 1 || scenario_idx == 3
        % pilot subcarriers
        pilot_indices_active = [1, 18, 35, 52]; % pilot range
        pilot_values = [0+0j, 0+1j, 1+0j, 1+1j];
        
        for sym_idx = 1:size(rx_symbols, 2)
            if sym_idx <= size(rx_symbols, 2)
                active_range = 1:N_active;
                rx_symbols(:, sym_idx) = simple_equalizer(rx_symbols(:, sym_idx), ...
                    pilot_indices_active', pilot_values', active_range');
            end
        end
    end
    
    % calculate evm
    rx_flat = rx_symbols(:);
    orig_flat = original_symbols(:);
    
    % remove invalid
    valid_idx = isfinite(rx_flat) & isfinite(orig_flat);
    rx_flat = rx_flat(valid_idx);
    orig_flat = orig_flat(valid_idx);
    
    if ~isempty(rx_flat) && ~isempty(orig_flat)
        % normalize power
        rx_normalized = rx_flat * sqrt(mean(abs(orig_flat).^2)) / sqrt(mean(abs(rx_flat).^2));
        error_vector = orig_flat - rx_normalized;
        evm_results(scenario_idx) = sqrt(mean(abs(error_vector).^2)) / sqrt(mean(abs(orig_flat).^2)) * 100;
        
        % plot constellation
        subplot(2,3,scenario_idx+1);
        scatter(real(rx_normalized), imag(rx_normalized), 20, colors{scenario_idx}, 'filled');
        title(sprintf('%s\nEVM: %.2f%%', scenario_names{scenario_idx}, evm_results(scenario_idx)));
        xlabel('In-phase'); ylabel('Quadrature');
        grid on; axis equal; axis([-2 2 -2 2]);
    else
        evm_results(scenario_idx) = NaN;
    end
end

%% delay sweep
fprintf('\n--- Delay Sweep Analysis ---\n');
delay_range = 0:delay_step:max_delay_samples*2;  % test range
evm_vs_delay = zeros(size(delay_range));

for delay_idx = 1:length(delay_range)
    delay = delay_range(delay_idx);
    gain = 0.7 * exp(1j * pi/4);  % fixed values
    
    % apply multipath
    channel_output = multipath_channel(tx_signal, [0, delay], [1, gain]);
    
    % demodulate symbols
    N_test = min(5, N_test_symbols);
    rx_syms = zeros(N_active, N_test);
    
    for sym_idx = 1:N_test
        start_idx = (sym_idx-1) * (N_fft + N_guard) + 1;
        end_idx = start_idx + N_fft + N_guard - 1;
        
        if end_idx <= length(channel_output)
            rx_symbol = channel_output(start_idx:end_idx);
            rx_useful = rx_symbol(N_guard+1:end);
            freq_rx = fft(rx_useful, N_fft);
            active_indices = (-N_active/2:N_active/2-1) + N_fft/2 + 1;
            rx_syms(:, sym_idx) = freq_rx(active_indices);
        end
    end
    
    % calculate evm
    rx_test = rx_syms(:);
    orig_test = original_symbols(:, 1:N_test);
    orig_test = orig_test(:);
    
    valid_idx = isfinite(rx_test) & isfinite(orig_test);
    if sum(valid_idx) > 0
        rx_test = rx_test(valid_idx);
        orig_test = orig_test(valid_idx);
        
        rx_norm = rx_test * sqrt(mean(abs(orig_test).^2)) / sqrt(mean(abs(rx_test).^2));
        evm_vs_delay(delay_idx) = sqrt(mean(abs(orig_test - rx_norm).^2)) / sqrt(mean(abs(orig_test).^2)) * 100;
    else
        evm_vs_delay(delay_idx) = NaN;
    end
end

% plot evm
subplot(2,3,5);
plot(delay_range / fs * 1e6, evm_vs_delay, 'b-o', 'LineWidth', 2);
hold on;
guard_time_us = T_guard_samples / fs * 1e6;
plot([guard_time_us, guard_time_us], ylim, 'r--', 'LineWidth', 2);
title('EVM vs Multipath Delay');
xlabel('Delay (μs)');
ylabel('EVM (%)');
legend('EVM', 'Guard Time', 'Location', 'northwest');
grid on;

% plot effectiveness
subplot(2,3,6);
delay_normalized = delay_range / T_guard_samples;
plot(delay_normalized, evm_vs_delay, 'b-o', 'LineWidth', 2);
hold on;
plot([1, 1], ylim, 'r--', 'LineWidth', 2);
title('EVM vs Normalized Delay');
xlabel('Delay / Guard Time');
ylabel('EVM (%)');
legend('EVM', 'Guard Boundary', 'Location', 'northwest');
grid on;

%% results summary
fprintf('\n=== Results Summary ===\n');
for i = 1:length(scenarios)
    if ~isnan(evm_results(i))
        fprintf('%s: EVM = %.2f%%\n', scenario_names{i}, evm_results(i));
    end
end

% find tolerance
tolerance_evm = 10;  % evm threshold
within_tolerance = delay_range(evm_vs_delay < tolerance_evm);
if ~isempty(within_tolerance)
    max_tolerable_delay = max(within_tolerance);
    fprintf('\nMaximum tolerable delay (EVM < %.0f%%): %d samples (%.2f μs)\n', ...
            tolerance_evm, max_tolerable_delay, max_tolerable_delay / fs * 1e6);
    fprintf('Guard time utilization: %.1f%%\n', max_tolerable_delay / T_guard_samples * 100);
else
    fprintf('\nNo delays meet the EVM threshold of %.0f%%\n', tolerance_evm);
end

% theoretical vs practical
fprintf('\nTheoretical guard time: %d samples (%.2f μs)\n', T_guard_samples, guard_time_us);
fprintf('Practical delay tolerance: depends on channel conditions and EVM requirements\n');

%% save results
save('exercise_4_1_3_results.mat', 'evm_vs_delay', 'delay_range', 'T_guard_samples', ...
     'evm_results', 'scenario_names', 'tolerance_evm');

% save figure
saveas(gcf, 'exercise_4_1_3_results.png');
fprintf('\nFigure saved as exercise_4_1_3_results.png\n');

fprintf('\nExercise 4.1.3 completed successfully!\n');
fprintf('Key finding: OFDM with cyclic prefix can handle multipath delays up to the guard time duration.\n');