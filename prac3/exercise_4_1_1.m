%% COMS4105 Practical 3 - Exercise 4.1.1
% ofdm mod/demod implementation
% 52 sub-carriers, 2mhz, qpsk

clear; close all; clc;

%% system parameters
N_active = 52;           % active sub-carriers
bandwidth = 2e6;         % 2mhz total
T_symbol = 36e-6;        % symbol duration
guard_ratio = 1/8;       % guard ratio

% calculate derived
T_useful = T_symbol * 8/9;  % fft period
T_guard = T_symbol - T_useful;  % guard time
subcarrier_spacing = 1/T_useful;  % spacing

% fft size calc
% fit 52 carriers in 2mhz
N_fft = round(bandwidth / subcarrier_spacing);  % fft size
if mod(N_fft, 2) ~= 0
    N_fft = N_fft + 1;  % even fft
end

N_guard = round(N_fft * guard_ratio);  % guard samples
fs = N_fft / T_useful;  % sampling freq

fprintf('=== OFDM System Parameters ===\n');
fprintf('FFT Size: %d\n', N_fft);
fprintf('Active Subcarriers: %d\n', N_active);
fprintf('Sampling Frequency: %.2f MHz\n', fs/1e6);
fprintf('Subcarrier Spacing: %.2f kHz\n', subcarrier_spacing/1000);
fprintf('Guard Samples: %d\n', N_guard);
fprintf('Useful Symbol Period: %.1f μs\n', T_useful*1e6);
fprintf('Guard Period: %.1f μs\n', T_guard*1e6);

%% pilot assignment
% pilots at indices
pilot_indices = [1, 18, 35, 52]; % matlab indexing
pilot_symbols = [0+0j, 0+1j, 1+0j, 1+1j]; % qpsk

%% data generation
N_data = N_active - length(pilot_indices);  % data subcarriers
N_symbols = 10;  % ofdm symbols

% generate qpsk data
data_bits = randi([0 1], N_data * 2, N_symbols);  % bits
qpsk_constellation = [1+1j, -1+1j, 1-1j, -1-1j]/sqrt(2);  % qpsk

% map bits to qpsk
data_symbols = zeros(N_data, N_symbols);
for sym_idx = 1:N_symbols
    for data_idx = 1:N_data
        bit_pair = data_bits((data_idx-1)*2+1:(data_idx-1)*2+2, sym_idx);
        % qpsk mapping
        if bit_pair(1) == 0 && bit_pair(2) == 0
            data_symbols(data_idx, sym_idx) = qpsk_constellation(1);
        elseif bit_pair(1) == 0 && bit_pair(2) == 1
            data_symbols(data_idx, sym_idx) = qpsk_constellation(3);
        elseif bit_pair(1) == 1 && bit_pair(2) == 0
            data_symbols(data_idx, sym_idx) = qpsk_constellation(2);
        else
            data_symbols(data_idx, sym_idx) = qpsk_constellation(4);
        end
    end
end

%% ofdm modulator
ofdm_signal = zeros(N_fft + N_guard, N_symbols);

for sym_idx = 1:N_symbols
    % frequency domain
    freq_domain = zeros(N_fft, 1);
    
    % map active carriers
    active_indices = (-N_active/2:N_active/2-1) + N_fft/2 + 1;
    
    % insert pilots
    data_counter = 1;
    for i = 1:N_active
        if any(i == pilot_indices)
            pilot_idx = find(i == pilot_indices);
            freq_domain(active_indices(i)) = pilot_symbols(pilot_idx);
        else
            freq_domain(active_indices(i)) = data_symbols(data_counter, sym_idx);
            data_counter = data_counter + 1;
        end
    end
    
    % ifft to time
    time_domain = ifft(freq_domain, N_fft);
    
    % add cyclic prefix
    guard_part = time_domain(end-N_guard+1:end);
    ofdm_symbol = [guard_part; time_domain];
    
    ofdm_signal(:, sym_idx) = ofdm_symbol;
end

% concatenate symbols
tx_signal = ofdm_signal(:);

fprintf('\n=== Modulation Complete ===\n');
fprintf('Generated %d OFDM symbols\n', N_symbols);
fprintf('Total signal length: %d samples\n', length(tx_signal));

%% ofdm demodulator
rx_signal = tx_signal;  % perfect channel
rx_symbols = zeros(N_active, N_symbols);

for sym_idx = 1:N_symbols
    % extract symbol
    start_idx = (sym_idx-1) * (N_fft + N_guard) + 1;
    end_idx = start_idx + N_fft + N_guard - 1;
    rx_symbol = rx_signal(start_idx:end_idx);
    
    % remove cyclic prefix
    rx_useful = rx_symbol(N_guard+1:end);
    
    % fft to frequency
    freq_rx = fft(rx_useful, N_fft);
    
    % extract carriers
    active_indices = (-N_active/2:N_active/2-1) + N_fft/2 + 1;
    rx_symbols(:, sym_idx) = freq_rx(active_indices);
end

fprintf('=== Demodulation Complete ===\n');

%% plot results
figure(1);
subplot(2,2,1);
plot(real(tx_signal), 'b', 'LineWidth', 1);
hold on;
plot(imag(tx_signal), 'r', 'LineWidth', 1);
title('OFDM Time Domain Signal');
xlabel('Sample Index');
ylabel('Amplitude');
legend('Real', 'Imaginary');
grid on;

subplot(2,2,2);
% plot spectrum
first_symbol = ofdm_signal(N_guard+1:end, 1);
freq_spectrum = fft(first_symbol, N_fft);
freq_axis = (-N_fft/2:N_fft/2-1) * fs/N_fft / 1000;
plot(freq_axis, fftshift(abs(freq_spectrum)));
title('OFDM Frequency Spectrum');
xlabel('Frequency (kHz)');
ylabel('Magnitude');
grid on;

subplot(2,2,3);
% tx constellation
tx_data_flat = data_symbols(:);
scatter(real(tx_data_flat), imag(tx_data_flat), 'bo');
title('Transmitted QPSK Constellation');
xlabel('In-phase');
ylabel('Quadrature');
grid on;
axis equal;

subplot(2,2,4);
% rx constellation
rx_data = rx_symbols;
pilot_mask = false(N_active, 1);
pilot_mask(pilot_indices) = true;
data_mask = ~pilot_mask;
rx_data_only = rx_data(data_mask, :);
rx_data_flat = rx_data_only(:);

scatter(real(rx_data_flat), imag(rx_data_flat), 'ro');
title('Received QPSK Constellation');
xlabel('In-phase');
ylabel('Quadrature');
grid on;
axis equal;

%% performance analysis
% calculate evm
tx_data_flat = data_symbols(:);
rx_data_flat = rx_data_only(:);

% normalize rx power
rx_normalized = rx_data_flat * sqrt(mean(abs(tx_data_flat).^2)) / sqrt(mean(abs(rx_data_flat).^2));
error_vector = tx_data_flat - rx_normalized;
evm_rms = sqrt(mean(abs(error_vector).^2)) / sqrt(mean(abs(tx_data_flat).^2)) * 100;

fprintf('EVM (RMS): %.4f%%\n', evm_rms);

%% ber calculation
% demodulate qpsk to bits
rx_bits = zeros(size(data_bits));
for sym_idx = 1:N_symbols
    data_counter = 1;
    for i = 1:N_active
        if ~any(i == pilot_indices)
            rx_symbol = rx_data(i, sym_idx);
            
            % qpsk decision
            if real(rx_symbol) > 0 && imag(rx_symbol) > 0
                bit_pair = [0, 0];
            elseif real(rx_symbol) > 0 && imag(rx_symbol) < 0
                bit_pair = [0, 1];
            elseif real(rx_symbol) < 0 && imag(rx_symbol) > 0
                bit_pair = [1, 0];
            else
                bit_pair = [1, 1];
            end
            
            rx_bits((data_counter-1)*2+1:(data_counter-1)*2+2, sym_idx) = bit_pair;
            data_counter = data_counter + 1;
        end
    end
end

% calculate ber
bit_errors = sum(sum(data_bits ~= rx_bits));
total_bits = numel(data_bits);
ber = bit_errors / total_bits;

fprintf('Bit Error Rate: %.2e (%d errors out of %d bits)\n', ber, bit_errors, total_bits);

%% save results
save('exercise_4_1_1_results.mat', 'ofdm_signal', 'tx_signal', 'N_fft', 'N_guard', 'fs', ...
     'N_active', 'pilot_indices', 'pilot_symbols', 'subcarrier_spacing');

% save figure
saveas(gcf, 'exercise_4_1_1_results.png');
fprintf('\nFigure saved as exercise_4_1_1_results.png\n');
fprintf('\nExercise 4.1.1 completed successfully!\n');