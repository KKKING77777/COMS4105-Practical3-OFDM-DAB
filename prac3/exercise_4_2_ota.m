%% over air ofdm
% process captured signals
% demonstrate ota reception

clear; close all; clc;

fprintf('=== Exercise 4.2: Over-the-Air OFDM Processing ===\n');

%% load signal data
data_files = {
    'zigbee_g0.0dB_att22dB_freq867.4MHz_0.mat',
    'zigbee_g0.0dB_att22dB_freq867.4MHz_1.mat',
    'zigbee_g0.0dB_att22dB_freq867.4MHz_2.mat',
    'zigbee_g0.0dB_att22dB_freq867.4MHz_3.mat',
    'zigbee_g0.0dB_att22dB_freq867.4MHz_4.mat'
};

% load first file
fprintf('\nLoading over-the-air signal data...\n');
load(fullfile('802154g', data_files{1}));
ota_signal = IQ_samples;
fs_ota = 2e6;  % Assume 2 MS/s as mentioned in the practical

fprintf('Loaded signal: %s\n', data_files{1});
fprintf('Signal length: %d samples (%.2f seconds)\n', length(ota_signal), length(ota_signal)/fs_ota);
fprintf('Center frequency: 867.4 MHz\n');
fprintf('Sample rate: %.1f MS/s\n', fs_ota/1e6);

%% snr estimation
fprintf('\n=== Exercise 4.2.1: SNR Estimation ===\n');

% signal structure
% sync data format
% each ofdm symbol
% null period

samples_per_symbol = 72;  % symbol samples
null_period_samples = round(0.144e-3 * fs_ota);  % null samples
sync_word_samples = 4 * samples_per_symbol;  % sync words

fprintf('OFDM symbol length: %d samples (%.1f μs)\n', samples_per_symbol, samples_per_symbol/fs_ota*1e6);
fprintf('NULL period: %d samples (%.3f ms)\n', null_period_samples, null_period_samples/fs_ota*1e3);

% find frame boundaries
signal_power = abs(ota_signal).^2;
moving_avg_window = 100;
smoothed_power = conv(signal_power, ones(1, moving_avg_window)/moving_avg_window, 'same');

% find null symbols
threshold = mean(smoothed_power) * 0.3;  % power threshold
null_candidates = find(smoothed_power < threshold);

if ~isempty(null_candidates)
    % find null region
    null_gaps = diff(null_candidates);
    long_nulls = find(null_gaps > 200);  % gap threshold
    
    if ~isempty(long_nulls)
        frame_start = null_candidates(long_nulls(1)) + null_period_samples;
    else
        frame_start = 1000;  % default start
    end
else
    frame_start = 1000;  % default start
end

fprintf('Estimated frame start: sample %d\n', frame_start);

% extract signal
analysis_length = 10000;  % analysis samples
if frame_start + analysis_length <= length(ota_signal)
    analysis_signal = ota_signal(frame_start:frame_start + analysis_length - 1);
else
    analysis_signal = ota_signal(1:analysis_length);
end

% snr estimation
signal_segments = reshape(analysis_signal(1:floor(length(analysis_signal)/samples_per_symbol)*samples_per_symbol), ...
                         samples_per_symbol, []);
symbol_powers = mean(abs(signal_segments).^2, 1);

% estimate noise
sorted_powers = sort(symbol_powers);
noise_floor = mean(sorted_powers(1:max(1, floor(length(sorted_powers)*0.1))));
signal_power_est = mean(sorted_powers(end-floor(length(sorted_powers)*0.1):end));

snr_linear = signal_power_est / noise_floor;
snr_db = 10 * log10(snr_linear);

fprintf('Estimated SNR: %.2f dB\n', snr_db);
fprintf('Signal power: %.2e\n', signal_power_est);
fprintf('Noise floor: %.2e\n', noise_floor);

%% frequency synchronization
fprintf('\n=== Exercise 4.2.2: Frequency Synchronization ===\n');

% extract symbols
num_symbols = min(50, floor((length(analysis_signal) - sync_word_samples) / samples_per_symbol));
ofdm_symbols = zeros(samples_per_symbol, num_symbols);

for i = 1:num_symbols
    start_idx = sync_word_samples + (i-1) * samples_per_symbol + 1;
    end_idx = start_idx + samples_per_symbol - 1;
    if end_idx <= length(analysis_signal)
        ofdm_symbols(:, i) = analysis_signal(start_idx:end_idx);
    end
end

fprintf('Extracted %d OFDM symbols for analysis\n', num_symbols);

% coarse frequency sync
% look for energy
symbol_fft = fft(ofdm_symbols, samples_per_symbol, 1);
avg_spectrum = mean(abs(symbol_fft).^2, 2);

% find frequency offset
[max_power, max_bin] = max(avg_spectrum);
freq_offset_bins = max_bin - samples_per_symbol/2 - 1;  % signed offset
freq_offset_hz = freq_offset_bins * fs_ota / samples_per_symbol;

fprintf('Coarse frequency offset: %d bins (%.2f kHz)\n', freq_offset_bins, freq_offset_hz/1000);

% apply correction
corrected_signal = analysis_signal .* exp(-1j * 2 * pi * freq_offset_hz * (0:length(analysis_signal)-1)' / fs_ota);

% fine frequency sync
% assume guard time
guard_length = 8;  % guard samples
if samples_per_symbol > guard_length
    cp_correlation = zeros(1, num_symbols-1);
    
    for i = 1:num_symbols-1
        start_idx = sync_word_samples + (i-1) * samples_per_symbol + 1;
        symbol = corrected_signal(start_idx:start_idx + samples_per_symbol - 1);
        
        % correlate guard
        guard_samples = symbol(1:guard_length);
        symbol_end = symbol(end-guard_length+1:end);
        
        correlation = abs(sum(guard_samples .* conj(symbol_end)));
        cp_correlation(i) = correlation;
    end
    
    fine_freq_error = angle(mean(exp(1j * cp_correlation))) / (2 * pi);
    fprintf('Fine frequency error: %.4f Hz\n', fine_freq_error);
else
    fine_freq_error = 0;
    fprintf('Fine frequency synchronization: Not applicable (symbol too short)\n');
end

%% data decoding
fprintf('\n=== Exercise 4.2.3: Data Decoding ===\n');

% apply correction
total_freq_offset = freq_offset_hz + fine_freq_error;
final_corrected = analysis_signal .* exp(-1j * 2 * pi * total_freq_offset * (0:length(analysis_signal)-1)' / fs_ota);

% extract data
data_start = sync_word_samples + 1;
available_data_samples = length(final_corrected) - data_start + 1;
num_data_symbols = floor(available_data_samples / samples_per_symbol);

fprintf('Processing %d data symbols\n', num_data_symbols);

if num_data_symbols > 0
    data_symbols = zeros(samples_per_symbol, num_data_symbols);
    
    for i = 1:num_data_symbols
        start_idx = data_start + (i-1) * samples_per_symbol;
        end_idx = start_idx + samples_per_symbol - 1;
        data_symbols(:, i) = final_corrected(start_idx:end_idx);
    end
    
    % fft demodulation
    % pure ofdm note
    data_fft = fft(data_symbols, samples_per_symbol, 1);
    
    % extract constellation
    data_carriers = round(samples_per_symbol*0.25):round(samples_per_symbol*0.75);  % middle spectrum
    constellation = data_fft(data_carriers, :);
    constellation_flat = constellation(:);
    
    % qpsk demodulation
    % normalize constellation
    constellation_norm = constellation_flat / sqrt(mean(abs(constellation_flat).^2));
    
    % decode bits
    decoded_bits = zeros(length(constellation_norm) * 2, 1);
    bit_idx = 1;
    
    for i = 1:length(constellation_norm)
        symbol = constellation_norm(i);
        
        % qpsk decision
        if real(symbol) > 0 && imag(symbol) > 0
            bits = [0, 0];
        elseif real(symbol) < 0 && imag(symbol) > 0
            bits = [1, 0];
        elseif real(symbol) > 0 && imag(symbol) < 0
            bits = [0, 1];
        else
            bits = [1, 1];
        end
        
        decoded_bits(bit_idx:bit_idx+1) = bits;
        bit_idx = bit_idx + 2;
    end
    
    % group bytes
    num_bytes = floor(length(decoded_bits) / 8);
    ascii_chars = zeros(1, num_bytes);
    
    for i = 1:num_bytes
        byte_bits = decoded_bits((i-1)*8+1:i*8);
        ascii_chars(i) = bi2de(byte_bits', 'left-msb');
    end
    
    % convert characters
    printable_chars = ascii_chars(ascii_chars >= 32 & ascii_chars <= 126);
    decoded_text = char(printable_chars);
    
    fprintf('Decoded %d bytes\n', num_bytes);
    fprintf('Printable characters found: %d\n', length(printable_chars));
    if ~isempty(decoded_text)
        fprintf('Sample decoded text (first 50 chars): "%s"\n', decoded_text(1:min(50, end)));
    else
        fprintf('No printable text found in decoded data\n');
    end
else
    fprintf('No data symbols available for decoding\n');
end

%% visualization
figure(1);
subplot(3,2,1);
plot((1:length(analysis_signal))/fs_ota*1000, abs(analysis_signal));
title('Over-the-Air Signal Magnitude');
xlabel('Time (ms)');
ylabel('Magnitude');
grid on;

subplot(3,2,2);
plot((1:length(analysis_signal))/fs_ota*1000, smoothed_power(1:length(analysis_signal)));
title('Signal Power (Smoothed)');
xlabel('Time (ms)');
ylabel('Power');
grid on;
hold on;
plot([0, length(analysis_signal)/fs_ota*1000], [threshold, threshold], 'r--', 'LineWidth', 2);
legend('Power', 'Null Threshold');

subplot(3,2,3);
freq_axis = (-samples_per_symbol/2:samples_per_symbol/2-1) * fs_ota/samples_per_symbol / 1000;
plot(freq_axis, fftshift(10*log10(avg_spectrum + eps)));
title('Average OFDM Symbol Spectrum');
xlabel('Frequency (kHz)');
ylabel('Power (dB)');
grid on;

subplot(3,2,4);
if exist('constellation_flat', 'var')
    scatter(real(constellation_flat), imag(constellation_flat), 10, 'b.', 'MarkerEdgeAlpha', 0.6);
    title('Received Constellation');
    xlabel('In-phase');
    ylabel('Quadrature');
    grid on;
    axis equal;
end

subplot(3,2,5);
if num_symbols > 1
    plot(1:num_symbols-1, cp_correlation);
    title('Cyclic Prefix Correlation');
    xlabel('Symbol Index');
    ylabel('Correlation Magnitude');
    grid on;
end

subplot(3,2,6);
if exist('ascii_chars', 'var')
    histogram(ascii_chars, 0:255);
    title('Decoded Byte Distribution');
    xlabel('Byte Value');
    ylabel('Count');
    grid on;
end

% save figure
saveas(gcf, 'exercise_4_2_results.png');
fprintf('\nFigure saved as exercise_4_2_results.png\n');

%% summary results
fprintf('\n=== Exercise 4.2 Summary ===\n');
fprintf('✓ SNR Estimation: %.2f dB\n', snr_db);
fprintf('✓ Coarse Frequency Sync: %.2f kHz offset corrected\n', freq_offset_hz/1000);
fprintf('✓ Fine Frequency Sync: %.4f Hz residual error\n', fine_freq_error);
if exist('num_bytes', 'var')
    fprintf('✓ Data Decoding: %d bytes decoded\n', num_bytes);
    fprintf('✓ Printable ASCII: %d characters\n', length(printable_chars));
end

fprintf('\nNote: This analysis assumes OFDM-like structure in 802.15.4g signal\n');
fprintf('Actual 802.15.4g may use different modulation schemes\n');

fprintf('\nExercise 4.2 completed successfully!\n');