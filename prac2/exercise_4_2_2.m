%% COMS4105 Practical 2 - Exercise 4.2.2
% hamming-decoded text message extraction
% over-the-air signal processing with hamming decoding

clear; close all; clc;

%% load hamming decoder from exercise 4.1.1
% hamming (15,11) parameters
n = 15; k = 11;

% parity check matrix
H = [0 1 1 1 1 0 1 0 1 0 0 1 0 0 0;
     1 0 1 1 1 1 0 1 0 1 0 0 1 0 0;
     1 1 0 1 1 1 1 0 1 0 1 0 0 1 0;
     1 1 1 0 1 1 1 1 0 1 0 0 0 0 1];

fprintf('=== Hamming-Decoded Text Extraction ===\n');
fprintf('Using Hamming (15,11) decoder\n');

%% hamming decoder function
function [decoded, error_detected] = hamming_decode(received, H, k)
    n = length(received);
    
    % calculate syndrome
    syndrome = mod(received * H', 2);
    
    % check for errors
    if sum(syndrome) == 0
        % no error detected
        decoded = received(1:k);
        error_detected = 0;
    else
        % error detected, find error position
        syndrome_decimal = bi2de(syndrome, 'left-msb');
        
        if syndrome_decimal <= n && syndrome_decimal > 0
            % single error - correct it
            corrected = received;
            corrected(syndrome_decimal) = mod(corrected(syndrome_decimal) + 1, 2);
            decoded = corrected(1:k);
            error_detected = 1;
        else
            % multiple errors detected but cannot correct
            decoded = received(1:k);
            error_detected = 2;
        end
    end
end

%% signal parameters
fs = 2e6; % 2 MS/s sample rate

% packet structure from exercise 4.2.1
sync_samples = 128;
null_samples = 160;
training_samples = 256;
data_samples = 800;

% modulation parameters
samples_per_symbol = 16; % from specification

fprintf('Sample rate: %.1f MS/s\n', fs/1e6);
fprintf('Samples per symbol: %d\n', samples_per_symbol);

%% simulate received signal with hamming-encoded text
% create test message
test_text = 'HELLO WORLD COMS4105 PRACTICAL TWO FREQUENCY SYNC TEST';
fprintf('\nOriginal text: "%s"\n', test_text);

% convert text to bits (ascii)
ascii_values = double(test_text);
text_bits = [];
for char_val = ascii_values
    char_bits = de2bi(char_val, 8, 'left-msb');
    text_bits = [text_bits char_bits];
end

fprintf('Text length: %d characters (%d bits)\n', length(test_text), length(text_bits));

%% hamming encoding of text
% pad to multiple of k bits
padding_needed = mod(k - mod(length(text_bits), k), k);
if padding_needed > 0
    padded_bits = [text_bits zeros(1, padding_needed)];
else
    padded_bits = text_bits;
end

num_blocks = length(padded_bits) / k;
encoded_blocks = [];

fprintf('Padding needed: %d bits\n', padding_needed);
fprintf('Number of Hamming blocks: %d\n', num_blocks);

% hamming generator matrix
G = [1 0 0 0 0 0 0 0 0 0 0 0 1 1 1;
     0 1 0 0 0 0 0 0 0 0 0 1 0 1 1;
     0 0 1 0 0 0 0 0 0 0 0 1 1 0 1;
     0 0 0 1 0 0 0 0 0 0 0 1 1 1 0;
     0 0 0 0 1 0 0 0 0 0 0 1 1 1 1;
     0 0 0 0 0 1 0 0 0 0 0 0 1 1 1;
     0 0 0 0 0 0 1 0 0 0 0 1 0 0 1;
     0 0 0 0 0 0 0 1 0 0 0 0 1 0 1;
     0 0 0 0 0 0 0 0 1 0 0 1 1 0 0;
     0 0 0 0 0 0 0 0 0 1 0 0 1 1 0;
     0 0 0 0 0 0 0 0 0 0 1 0 0 1 1];

for block_idx = 1:num_blocks
    start_idx = (block_idx - 1) * k + 1;
    end_idx = block_idx * k;
    message_block = padded_bits(start_idx:end_idx);
    
    % hamming encoding
    codeword = mod(message_block * G, 2);
    encoded_blocks = [encoded_blocks codeword];
end

fprintf('Encoded length: %d bits (rate = %.3f)\n', length(encoded_blocks), length(padded_bits)/length(encoded_blocks));

%% create packet structure
% sync word: 10101011 (bpsk)
sync_bits = [1 0 1 0 1 0 1 1];

% training sequence (qpsk): 11110011 10100000 (repeated twice)
training_bits = [1 1 1 1 0 0 1 1 1 0 1 0 0 0 0 0];
training_repeated = [training_bits training_bits];

% create full packet
packet_bits = [sync_bits zeros(1, 16) training_repeated encoded_blocks];

% truncate to fit data samples constraint
max_data_bits = data_samples / samples_per_symbol * 2; % qpsk = 2 bits/symbol
if length(encoded_blocks) > max_data_bits
    encoded_blocks = encoded_blocks(1:max_data_bits);
    fprintf('Data truncated to %d bits to fit packet structure\n', max_data_bits);
end

%% modulate signal
% sync: bpsk modulation
sync_symbols = 2 * sync_bits - 1;

% training: qpsk modulation
qpsk_map = [1+1j, 1-1j, -1+1j, -1-1j]/sqrt(2); % 00,01,10,11
training_symbols = [];
for i = 1:2:length(training_repeated)
    bit_pair = training_repeated(i:i+1);
    symbol_idx = bi2de(bit_pair, 'left-msb') + 1;
    training_symbols = [training_symbols qpsk_map(symbol_idx)];
end

% data: qpsk modulation
data_symbols = [];
for i = 1:2:length(encoded_blocks)
    if i+1 <= length(encoded_blocks)
        bit_pair = encoded_blocks(i:i+1);
        symbol_idx = bi2de(bit_pair, 'left-msb') + 1;
        data_symbols = [data_symbols qpsk_map(symbol_idx)];
    end
end

%% oversample and create signal
sync_signal = repelem(sync_symbols, samples_per_symbol);
null_signal = zeros(1, null_samples);
training_signal = repelem(training_symbols, samples_per_symbol);
data_signal = repelem(data_symbols, samples_per_symbol);

% pad data to correct length
if length(data_signal) < data_samples
    data_signal = [data_signal zeros(1, data_samples - length(data_signal))];
elseif length(data_signal) > data_samples  
    data_signal = data_signal(1:data_samples);
end

% complete packet
complete_signal = [sync_signal null_signal training_signal data_signal];

fprintf('\nSignal lengths:\n');
fprintf('Sync: %d samples\n', length(sync_signal));
fprintf('Null: %d samples\n', length(null_signal));
fprintf('Training: %d samples\n', length(training_signal));
fprintf('Data: %d samples\n', length(data_signal));
fprintf('Total: %d samples\n', length(complete_signal));

%% add channel effects
snr_db = 12; % moderate snr
freq_offset_hz = 800; % frequency offset

% apply frequency offset
t = (0:length(complete_signal)-1) / fs;
tx_signal = complete_signal .* exp(1j * 2 * pi * freq_offset_hz * t);

% add awgn
noise_power = 10^(-snr_db/10);
noise = sqrt(noise_power/2) * (randn(size(tx_signal)) + 1j*randn(size(tx_signal)));
rx_signal = tx_signal + noise;

fprintf('\nChannel conditions:\n');
fprintf('SNR: %d dB\n', snr_db);
fprintf('Frequency offset: %d Hz\n', freq_offset_hz);

%% frequency synchronization (using method from 4.2.1)
% extract training sequences
training_start = sync_samples + null_samples + 1;
training1_end = training_start + training_samples/2 - 1;
training2_start = training1_end + 1;
training2_end = training2_start + training_samples/2 - 1;

rx_training1 = rx_signal(training_start:training1_end);
rx_training2 = rx_signal(training2_start:training2_end);

% estimate frequency offset (simplified)
phase_diff = angle(sum(conj(rx_training1) .* rx_training2));
time_sep = (training_samples/2) / fs;
estimated_offset = phase_diff / (2 * pi * time_sep);

fprintf('Estimated frequency offset: %.0f Hz (error: %.0f Hz)\n', ...
        estimated_offset, abs(freq_offset_hz - estimated_offset));

% apply frequency correction
correction_signal = exp(-1j * 2 * pi * estimated_offset * t);
corrected_signal = rx_signal .* correction_signal;

%% extract and demodulate data
data_start = sync_samples + null_samples + training_samples + 1;
data_end = data_start + data_samples - 1;
rx_data = corrected_signal(data_start:data_end);

% qpsk demodulation
demod_bits = [];
for i = 1:samples_per_symbol:length(rx_data)
    if i + samples_per_symbol - 1 <= length(rx_data)
        symbol = mean(rx_data(i:i+samples_per_symbol-1)); % integrate over symbol
        
        % qpsk decision regions
        if real(symbol) > 0 && imag(symbol) > 0
            bits = [0 0]; % 1+1j
        elseif real(symbol) > 0 && imag(symbol) < 0
            bits = [0 1]; % 1-1j
        elseif real(symbol) < 0 && imag(symbol) > 0
            bits = [1 0]; % -1+1j
        else
            bits = [1 1]; % -1-1j
        end
        demod_bits = [demod_bits bits];
    end
end

% truncate to encoded length
if length(demod_bits) > length(encoded_blocks)
    demod_bits = demod_bits(1:length(encoded_blocks));
end

fprintf('\nDemodulated %d bits\n', length(demod_bits));

%% hamming decoding
decoded_message = [];
bit_errors_before = sum(encoded_blocks ~= demod_bits);
corrected_errors = 0;

for block_idx = 1:num_blocks
    if (block_idx * n) <= length(demod_bits)
        start_idx = (block_idx - 1) * n + 1;
        end_idx = block_idx * n;
        received_block = demod_bits(start_idx:end_idx);
        
        [decoded_block, error_flag] = hamming_decode(received_block, H, k);
        decoded_message = [decoded_message decoded_block];
        
        if error_flag == 1
            corrected_errors = corrected_errors + 1;
        end
    end
end

fprintf('Hamming blocks processed: %d\n', min(num_blocks, floor(length(demod_bits)/n)));
fprintf('Bit errors before decoding: %d\n', bit_errors_before);
fprintf('Blocks with corrected errors: %d\n', corrected_errors);

%% convert back to text
% remove padding
decoded_text_bits = decoded_message(1:length(text_bits));
decoded_text = '';

for i = 1:8:length(decoded_text_bits)
    if i+7 <= length(decoded_text_bits)
        char_bits = decoded_text_bits(i:i+7);
        ascii_val = bi2de(char_bits, 'left-msb');
        if ascii_val >= 32 && ascii_val <= 126 % printable ascii
            decoded_text = [decoded_text char(ascii_val)];
        else
            decoded_text = [decoded_text '?']; % non-printable
        end
    end
end

fprintf('\nDecoded text: "%s"\n', decoded_text);

% calculate character error rate
char_errors = 0;
min_length = min(length(test_text), length(decoded_text));
for i = 1:min_length
    if test_text(i) ~= decoded_text(i)
        char_errors = char_errors + 1;
    end
end

cer = char_errors / length(test_text) * 100;
fprintf('Character Error Rate: %.1f%% (%d/%d)\n', cer, char_errors, length(test_text));

%% plot results
figure(1);
subplot(2,2,1);
plot(real(complete_signal), 'b', 'LineWidth', 1);
hold on;
plot(imag(complete_signal), 'r', 'LineWidth', 1);
title('Transmitted Signal');
xlabel('Sample');
ylabel('Amplitude');
legend('Real', 'Imaginary');
grid on;

subplot(2,2,2);
plot(real(rx_signal), 'b', 'LineWidth', 1);
hold on;
plot(imag(rx_signal), 'r', 'LineWidth', 1);
title('Received Signal (with offset and noise)');
xlabel('Sample');
ylabel('Amplitude');
legend('Real', 'Imaginary');
grid on;

subplot(2,2,3);
plot(real(corrected_signal), 'b', 'LineWidth', 1);
hold on;
plot(imag(corrected_signal), 'r', 'LineWidth', 1);
title('Frequency Corrected Signal');
xlabel('Sample');
ylabel('Amplitude');
legend('Real', 'Imaginary');
grid on;

subplot(2,2,4);
% constellation plot of data symbols
data_symbols_rx = rx_data(1:samples_per_symbol:end);
scatter(real(data_symbols_rx), imag(data_symbols_rx), 'bo');
hold on;
% overlay ideal qpsk constellation
ideal_qpsk = [1+1j, 1-1j, -1+1j, -1-1j]/sqrt(2);
scatter(real(ideal_qpsk), imag(ideal_qpsk), 'rx', 'LineWidth', 2, 'SizeData', 100);
title('QPSK Constellation');
xlabel('In-phase');
ylabel('Quadrature');
legend('Received', 'Ideal');
grid on;
axis equal;

%% save results
save('exercise_4_2_2_results.mat', 'test_text', 'decoded_text', 'cer', ...
     'bit_errors_before', 'corrected_errors');

saveas(gcf, 'exercise_4_2_2_text_decoding.png');

fprintf('\nExercise 4.2.2 completed successfully!\n');