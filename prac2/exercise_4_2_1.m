%% COMS4105 Practical 2 - Exercise 4.2.1
% frequency synchronization with over-the-air signals
% least squares frequency estimation

clear; close all; clc;

%% signal parameters (from practical specification)
fs = 2e6;        % sample rate 2 MS/s
fc_freq_a = 924e6; % frequency A (example)

% packet structure parameters
sync_duration = 0.064e-3;    % 64 μs sync
training_duration = 0.128e-3; % 128 μs training (64μs x 2)
data_duration = 0.4e-3;      % 400 μs data

sync_samples = round(sync_duration * fs);       % 128 samples
training_samples = round(training_duration * fs); % 256 samples  
data_samples = round(data_duration * fs);       % 800 samples

fprintf('=== Frequency Synchronization Parameters ===\n');
fprintf('Sample rate: %.1f MS/s\n', fs/1e6);
fprintf('Sync samples: %d\n', sync_samples);
fprintf('Training samples: %d (2 x %d)\n', training_samples, training_samples/2);
fprintf('Data samples: %d\n', data_samples);

%% generate test signal with known frequency offset
% sync word: 10101011 (bpsk)
sync_bits = [1 0 1 0 1 0 1 1];
sync_symbols = 2 * sync_bits - 1; % bpsk mapping

% training sequence: 11110011 10100000 (repeated twice, qpsk)
training_bits1 = [1 1 1 1 0 0 1 1 1 0 1 0 0 0 0 0];
training_bits2 = training_bits1; % repeated

% convert to qpsk symbols
qpsk_map = [1+1j, 1-1j, -1+1j, -1-1j]/sqrt(2); % 00,01,10,11
training_symbols1 = zeros(1, length(training_bits1)/2);
training_symbols2 = zeros(1, length(training_bits2)/2);

for i = 1:2:length(training_bits1)
    bit_pair = training_bits1(i:i+1);
    symbol_idx = bi2de(bit_pair, 'left-msb') + 1;
    training_symbols1((i+1)/2) = qpsk_map(symbol_idx);
end

for i = 1:2:length(training_bits2)
    bit_pair = training_bits2(i:i+1);
    symbol_idx = bi2de(bit_pair, 'left-msb') + 1;
    training_symbols2((i+1)/2) = qpsk_map(symbol_idx);
end

% oversample to match sample rate
samples_per_symbol_sync = sync_samples / length(sync_symbols);
samples_per_symbol_training = (training_samples/2) / length(training_symbols1);

fprintf('\nSamples per symbol: Sync=%.1f, Training=%.1f\n', ...
        samples_per_symbol_sync, samples_per_symbol_training);

%% create oversampled signals
% sync signal (bpsk)
sync_signal = zeros(1, sync_samples);
for i = 1:length(sync_symbols)
    start_idx = round((i-1) * samples_per_symbol_sync) + 1;
    end_idx = min(round(i * samples_per_symbol_sync), sync_samples);
    sync_signal(start_idx:end_idx) = sync_symbols(i);
end

% training signals (qpsk)
training1_signal = zeros(1, training_samples/2);
for i = 1:length(training_symbols1)
    start_idx = round((i-1) * samples_per_symbol_training) + 1;
    end_idx = min(round(i * samples_per_symbol_training), training_samples/2);
    training1_signal(start_idx:end_idx) = training_symbols1(i);
end

training2_signal = zeros(1, training_samples/2);
for i = 1:length(training_symbols2)
    start_idx = round((i-1) * samples_per_symbol_training) + 1;
    end_idx = min(round(i * samples_per_symbol_training), training_samples/2);
    training2_signal(start_idx:end_idx) = training_symbols2(i);
end

%% frequency offset estimation function
function freq_offset_hz = estimate_freq_offset(preamble1, preamble2, fs)
    % least squares frequency offset estimation
    % using relationship: y[t+Npre] = exp(j2πfc*Npre) * y[t]
    
    if length(preamble1) ~= length(preamble2)
        error('Preambles must have same length');
    end
    
    % form least squares problem: y2 = exp(j2πfcT) * y1
    % where T = Npre/fs is the time between preambles
    
    % least squares solution for complex exponential
    numerator = sum(conj(preamble1) .* preamble2);
    denominator = sum(abs(preamble1).^2);
    
    if abs(denominator) < 1e-10
        freq_offset_hz = 0;
        return;
    end
    
    % estimated complex exponential
    complex_exp = numerator / denominator;
    
    % extract frequency from phase
    phase_shift = angle(complex_exp);
    time_separation = length(preamble1) / fs;
    
    freq_offset_hz = phase_shift / (2 * pi * time_separation);
    
    % handle phase wrapping ambiguity
    max_unambiguous_freq = fs / (2 * length(preamble1));
    while freq_offset_hz > max_unambiguous_freq
        freq_offset_hz = freq_offset_hz - 2 * max_unambiguous_freq;
    end
    while freq_offset_hz < -max_unambiguous_freq
        freq_offset_hz = freq_offset_hz + 2 * max_unambiguous_freq;
    end
end

%% test frequency estimation with various offsets
test_offsets_hz = [-2000, -500, 0, 500, 2000, 5000]; % test frequencies

fprintf('\n=== Frequency Estimation Test ===\n');
fprintf('Testing frequency offsets:\n');

estimated_offsets = zeros(size(test_offsets_hz));

for i = 1:length(test_offsets_hz)
    offset_hz = test_offsets_hz(i);
    
    % apply frequency offset
    t1 = (0:length(training1_signal)-1) / fs;
    t2 = (0:length(training2_signal)-1) / fs + length(training1_signal)/fs;
    
    offset1 = training1_signal .* exp(1j * 2 * pi * offset_hz * t1);
    offset2 = training2_signal .* exp(1j * 2 * pi * offset_hz * t2);
    
    % add some noise
    snr_db = 20;
    noise_power = 10^(-snr_db/10);
    noise1 = sqrt(noise_power/2) * (randn(size(offset1)) + 1j*randn(size(offset1)));
    noise2 = sqrt(noise_power/2) * (randn(size(offset2)) + 1j*randn(size(offset2)));
    
    noisy1 = offset1 + noise1;
    noisy2 = offset2 + noise2;
    
    % estimate frequency offset
    estimated_offsets(i) = estimate_freq_offset(noisy1, noisy2, fs);
    
    error_hz = abs(offset_hz - estimated_offsets(i));
    
    fprintf('True: %6.0f Hz, Estimated: %6.0f Hz, Error: %6.1f Hz\n', ...
            offset_hz, estimated_offsets(i), error_hz);
end

%% calculate theoretical limits
npre_samples = training_samples / 2;
max_unambiguous_freq = fs / (2 * npre_samples);

fprintf('\n=== Theoretical Limits ===\n');
fprintf('Preamble length: %d samples\n', npre_samples);
fprintf('Time separation: %.3f ms\n', npre_samples/fs * 1000);
fprintf('Maximum unambiguous frequency: ±%.1f Hz\n', max_unambiguous_freq);
fprintf('Frequency resolution: %.1f Hz\n', fs/npre_samples);

%% simulate realistic over-the-air scenario
fprintf('\n=== Realistic OTA Simulation ===\n');

% simulate received packet with unknown offset
true_offset = 1500; % 1.5 kHz offset
snr_db = 15;

% create complete packet
null_section = zeros(1, 160); % null period
complete_packet = [sync_signal, null_section, training1_signal, training2_signal];

% apply frequency offset and noise
t_packet = (0:length(complete_packet)-1) / fs;
tx_signal = complete_packet .* exp(1j * 2 * pi * true_offset * t_packet);

% add awgn
noise_power = 10^(-snr_db/10);
noise = sqrt(noise_power/2) * (randn(size(tx_signal)) + 1j*randn(size(tx_signal)));
rx_signal = tx_signal + noise;

% extract training sequences from received signal
sync_end = length(sync_signal);
null_end = sync_end + length(null_section);
train1_start = null_end + 1;
train1_end = train1_start + length(training1_signal) - 1;
train2_start = train1_end + 1;
train2_end = train2_start + length(training2_signal) - 1;

rx_training1 = rx_signal(train1_start:train1_end);
rx_training2 = rx_signal(train2_start:train2_end);

% estimate frequency offset
estimated_offset = estimate_freq_offset(rx_training1, rx_training2, fs);
estimation_error = abs(true_offset - estimated_offset);

fprintf('True offset: %.0f Hz\n', true_offset);
fprintf('Estimated offset: %.0f Hz\n', estimated_offset);
fprintf('Estimation error: %.1f Hz\n', estimation_error);
fprintf('SNR: %.0f dB\n', snr_db);

%% plot results
figure(1);
subplot(3,1,1);
plot(test_offsets_hz, estimated_offsets, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(test_offsets_hz, test_offsets_hz, 'r--', 'LineWidth', 1);
grid on;
xlabel('True Frequency Offset (Hz)');
ylabel('Estimated Offset (Hz)');
title('Frequency Estimation Accuracy');
legend('Estimated', 'Ideal', 'Location', 'northwest');

subplot(3,1,2);
estimation_errors = abs(test_offsets_hz - estimated_offsets);
stem(test_offsets_hz, estimation_errors, 'r', 'LineWidth', 2);
grid on;
xlabel('True Frequency Offset (Hz)');
ylabel('Estimation Error (Hz)');
title('Frequency Estimation Errors');

subplot(3,1,3);
t_plot = (0:length(complete_packet)-1) / fs * 1000; % time in ms
plot(t_plot, real(rx_signal), 'b', 'LineWidth', 1);
hold on;
plot(t_plot, imag(rx_signal), 'r', 'LineWidth', 1);
grid on;
xlabel('Time (ms)');
ylabel('Amplitude');
title('Received Signal with Frequency Offset');
legend('Real', 'Imaginary');

%% performance vs snr
snr_range = 0:2:30;
num_trials = 100;
mean_errors = zeros(size(snr_range));

fprintf('\n=== Performance vs SNR ===\n');

for snr_idx = 1:length(snr_range)
    snr_test = snr_range(snr_idx);
    errors = zeros(1, num_trials);
    
    for trial = 1:num_trials
        % random offset
        random_offset = (rand - 0.5) * 2000; % ±1000 Hz
        
        % apply offset and noise
        tx_test = [training1_signal, training2_signal];
        t_test = (0:length(tx_test)-1) / fs;
        tx_offset = tx_test .* exp(1j * 2 * pi * random_offset * t_test);
        
        noise_power = 10^(-snr_test/10);
        noise = sqrt(noise_power/2) * (randn(size(tx_offset)) + 1j*randn(size(tx_offset)));
        rx_test = tx_offset + noise;
        
        % extract preambles
        rx_pre1 = rx_test(1:length(training1_signal));
        rx_pre2 = rx_test(length(training1_signal)+1:end);
        
        % estimate
        est_offset = estimate_freq_offset(rx_pre1, rx_pre2, fs);
        errors(trial) = abs(random_offset - est_offset);
    end
    
    mean_errors(snr_idx) = mean(errors);
    fprintf('SNR = %2d dB: Mean error = %.1f Hz\n', snr_test, mean_errors(snr_idx));
end

figure(2);
semilogy(snr_range, mean_errors, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('SNR (dB)');
ylabel('Mean Frequency Error (Hz)');
title('Frequency Estimation Performance vs SNR');

%% save results
save('exercise_4_2_1_results.mat', 'test_offsets_hz', 'estimated_offsets', ...
     'snr_range', 'mean_errors', 'max_unambiguous_freq');

saveas(figure(1), 'exercise_4_2_1_freq_estimation.png');
saveas(figure(2), 'exercise_4_2_1_snr_performance.png');

fprintf('\nExercise 4.2.1 completed successfully!\n');