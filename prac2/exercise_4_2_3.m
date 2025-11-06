%% COMS4105 Practical 2 - Exercise 4.2.3
% hamming + crc-16 frame decoding
% combined error correction and detection

clear; close all; clc;

%% load functions from previous exercises
% hamming (15,11) parameters
n_ham = 15; k_ham = 11;

% hamming matrices (from exercise 4.1.1)
I = eye(k_ham);
P = [1 1 1 1; 1 1 1 0; 1 1 0 1; 1 0 1 1; 0 1 1 1;
     1 1 0 0; 1 0 1 0; 1 0 0 1; 0 1 1 0; 0 1 0 1; 0 0 1 1];
G_ham = [I P];
H_ham = [P' eye(n_ham-k_ham)];

% crc-16 parameters (dab polynomial)
crc_poly = [1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1];

fprintf('=== Hamming + CRC-16 Frame Decoding ===\n');
fprintf('Hamming: (%d,%d) code\n', n_ham, k_ham);
fprintf('CRC: 16-bit checksum\n');

%% helper functions
function crc = crc16_encode(message)
    % crc-16 encoder
    poly = [1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1];
    dividend = [message zeros(1, 16)];
    
    for i = 1:(length(message))
        if dividend(i) == 1
            for j = 1:length(poly)
                dividend(i + j - 1) = xor(dividend(i + j - 1), poly(j));
            end
        end
    end
    crc = dividend(end-15:end);
end

function error_detected = crc16_check(frame)
    % crc-16 checker
    poly = [1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1];
    dividend = frame;
    
    for i = 1:(length(frame) - 16)
        if dividend(i) == 1
            for j = 1:length(poly)
                if (i + j - 1) <= length(dividend)
                    dividend(i + j - 1) = xor(dividend(i + j - 1), poly(j));
                end
            end
        end
    end
    remainder = dividend(end-15:end);
    error_detected = any(remainder);
end

function [decoded, error_detected] = hamming_decode(received, H, k)
    % hamming decoder
    syndrome = mod(received * H', 2);
    syndrome_value = bi2de(syndrome, 'right-msb');
    
    if syndrome_value == 0
        decoded = received(1:k);
        error_detected = 0;
    else
        if syndrome_value <= length(received)
            corrected = received;
            corrected(syndrome_value) = mod(corrected(syndrome_value) + 1, 2);
            decoded = corrected(1:k);
            error_detected = 1;
        else
            decoded = received(1:k);
            error_detected = 2;
        end
    end
end

%% signal parameters (from specification)
fs = 2e6; % 2 MS/s
samples_per_symbol = 16;

% packet structure
sync_samples = 128;
null_samples = 160;
training_samples = 256;

fprintf('Sample rate: %.1f MS/s\n', fs/1e6);
fprintf('Samples per symbol: %d\n', samples_per_symbol);

%% create test message with frame structure
test_text = 'FRAME TEST WITH HAMMING AND CRC16 CODING';
fprintf('\nOriginal message: "%s"\n', test_text);

% convert to bits
ascii_vals = double(test_text);
message_bits = [];
for val = ascii_vals
    char_bits = de2bi(val, 8, 'left-msb');
    message_bits = [message_bits char_bits];
end

% add crc-16 to each frame
frame_size = 64; % message bits per frame
frames = [];
num_frames = ceil(length(message_bits) / frame_size);

fprintf('Message length: %d bits\n', length(message_bits));
fprintf('Frame size: %d bits + 16 CRC\n', frame_size);
fprintf('Number of frames: %d\n', num_frames);

for frame_idx = 1:num_frames
    start_idx = (frame_idx - 1) * frame_size + 1;
    end_idx = min(frame_idx * frame_size, length(message_bits));
    
    frame_data = message_bits(start_idx:end_idx);
    
    % pad to frame size if needed
    if length(frame_data) < frame_size
        frame_data = [frame_data zeros(1, frame_size - length(frame_data))];
    end
    
    % add crc-16
    crc_bits = crc16_encode(frame_data);
    frame_with_crc = [frame_data crc_bits];
    
    frames = [frames frame_with_crc];
end

fprintf('Total frame data: %d bits\n', length(frames));

%% hamming encode the frames
% pad to multiple of k_ham
padding = mod(k_ham - mod(length(frames), k_ham), k_ham);
if padding > 0
    padded_frames = [frames zeros(1, padding)];
else
    padded_frames = frames;
end

num_ham_blocks = length(padded_frames) / k_ham;
encoded_data = [];

for block_idx = 1:num_ham_blocks
    start_idx = (block_idx - 1) * k_ham + 1;
    end_idx = block_idx * k_ham;
    block_data = padded_frames(start_idx:end_idx);
    
    % hamming encode
    codeword = mod(block_data * G_ham, 2);
    encoded_data = [encoded_data codeword];
end

fprintf('Hamming blocks: %d\n', num_ham_blocks);
fprintf('Encoded length: %d bits\n', length(encoded_data));

%% create packet with training sequences
% sync: 10101011 (bpsk)
sync_bits = [1 0 1 0 1 0 1 1];

% training: 11110011 10100000 (qpsk, repeated)
training_bits = [1 1 1 1 0 0 1 1 1 0 1 0 0 0 0 0];
training_repeated = [training_bits training_bits];

%% modulation
% sync: bpsk
sync_symbols = 2 * sync_bits - 1;

% training: qpsk  
qpsk_map = [1+1j, 1-1j, -1+1j, -1-1j]/sqrt(2);
training_symbols = [];
for i = 1:2:length(training_repeated)
    bit_pair = training_repeated(i:i+1);
    idx = bi2de(bit_pair, 'left-msb') + 1;
    training_symbols = [training_symbols qpsk_map(idx)];
end

% data: qpsk
data_symbols = [];
for i = 1:2:length(encoded_data)
    if i+1 <= length(encoded_data)
        bit_pair = encoded_data(i:i+1);
        idx = bi2de(bit_pair, 'left-msb') + 1;
        data_symbols = [data_symbols qpsk_map(idx)];
    end
end

%% create oversampled signals
sync_signal = repelem(sync_symbols, samples_per_symbol);
null_signal = zeros(1, null_samples);
training_signal = repelem(training_symbols, samples_per_symbol);
data_signal = repelem(data_symbols, samples_per_symbol);

% complete packet
complete_signal = [sync_signal null_signal training_signal data_signal];

fprintf('\nSignal structure:\n');
fprintf('Sync: %d samples\n', length(sync_signal));
fprintf('Training: %d samples\n', length(training_signal));
fprintf('Data: %d samples\n', length(data_signal));

%% channel simulation
snr_db = 10;
freq_offset_hz = 1200;

% apply frequency offset
t = (0:length(complete_signal)-1) / fs;
tx_signal = complete_signal .* exp(1j * 2 * pi * freq_offset_hz * t);

% add noise
noise_power = 10^(-snr_db/10);
noise = sqrt(noise_power/2) * (randn(size(tx_signal)) + 1j*randn(size(tx_signal)));
rx_signal = tx_signal + noise;

fprintf('\nChannel: SNR = %d dB, Freq offset = %d Hz\n', snr_db, freq_offset_hz);

%% frequency synchronization
training_start = length(sync_signal) + length(null_signal) + 1;
training1_end = training_start + length(training_signal)/2 - 1;
training2_start = training1_end + 1;
training2_end = training2_start + length(training_signal)/2 - 1;

rx_training1 = rx_signal(training_start:training1_end);
rx_training2 = rx_signal(training2_start:training2_end);

% estimate frequency offset
phase_diff = angle(sum(conj(rx_training1) .* rx_training2));
time_sep = length(rx_training1) / fs;
estimated_offset = phase_diff / (2 * pi * time_sep);

fprintf('Estimated frequency offset: %.0f Hz (error: %.0f Hz)\n', ...
        estimated_offset, abs(freq_offset_hz - estimated_offset));

% apply correction
correction = exp(-1j * 2 * pi * estimated_offset * t);
corrected_signal = rx_signal .* correction;

%% extract and demodulate data
data_start = length(sync_signal) + length(null_signal) + length(training_signal) + 1;
rx_data = corrected_signal(data_start:data_start + length(data_signal) - 1);

% qpsk demodulation
demod_bits = [];
for i = 1:samples_per_symbol:length(rx_data)
    if i + samples_per_symbol - 1 <= length(rx_data)
        symbol = mean(rx_data(i:i+samples_per_symbol-1));
        
        % qpsk decision
        if real(symbol) > 0 && imag(symbol) > 0
            bits = [0 0];
        elseif real(symbol) > 0 && imag(symbol) < 0
            bits = [0 1];
        elseif real(symbol) < 0 && imag(symbol) > 0
            bits = [1 0];
        else
            bits = [1 1];
        end
        demod_bits = [demod_bits bits];
    end
end

% match length
demod_bits = demod_bits(1:min(length(demod_bits), length(encoded_data)));

%% hamming decoding
decoded_frames = [];
hamming_corrections = 0;

for block_idx = 1:num_ham_blocks
    if block_idx * n_ham <= length(demod_bits)
        start_idx = (block_idx - 1) * n_ham + 1;
        end_idx = block_idx * n_ham;
        received_block = demod_bits(start_idx:end_idx);
        
        [decoded_block, error_flag] = hamming_decode(received_block, H_ham, k_ham);
        decoded_frames = [decoded_frames decoded_block];
        
        if error_flag == 1
            hamming_corrections = hamming_corrections + 1;
        end
    end
end

% remove padding
decoded_frames = decoded_frames(1:length(frames));

fprintf('\nHamming decoding:\n');
fprintf('Blocks processed: %d\n', num_ham_blocks);
fprintf('Errors corrected: %d\n', hamming_corrections);

%% crc checking and frame extraction
recovered_message = [];
crc_errors = 0;
valid_frames = 0;

for frame_idx = 1:num_frames
    start_idx = (frame_idx - 1) * (frame_size + 16) + 1;
    end_idx = min(frame_idx * (frame_size + 16), length(decoded_frames));
    
    if end_idx - start_idx + 1 >= frame_size + 16
        frame_data = decoded_frames(start_idx:end_idx);
        
        % check crc
        if ~crc16_check(frame_data)
            % crc ok, extract message
            message_part = frame_data(1:frame_size);
            recovered_message = [recovered_message message_part];
            valid_frames = valid_frames + 1;
        else
            crc_errors = crc_errors + 1;
            % still extract message (crc failed)
            message_part = frame_data(1:frame_size);
            recovered_message = [recovered_message message_part];
        end
    end
end

fprintf('\nCRC checking:\n');
fprintf('Valid frames: %d/%d\n', valid_frames, num_frames);
fprintf('CRC errors: %d\n', crc_errors);

%% convert back to text
recovered_text = '';
for i = 1:8:min(length(recovered_message), length(message_bits))
    if i+7 <= length(recovered_message)
        char_bits = recovered_message(i:i+7);
        ascii_val = bi2de(char_bits, 'left-msb');
        if ascii_val >= 32 && ascii_val <= 126
            recovered_text = [recovered_text char(ascii_val)];
        else
            recovered_text = [recovered_text '?'];
        end
    end
end

fprintf('\nRecovered text: "%s"\n', recovered_text);

% calculate performance
bit_errors = sum(message_bits(1:min(end,length(recovered_message))) ~= ...
                recovered_message(1:min(end,length(message_bits))));
ber = bit_errors / length(message_bits) * 100;

char_errors = 0;
min_len = min(length(test_text), length(recovered_text));
for i = 1:min_len
    if test_text(i) ~= recovered_text(i)
        char_errors = char_errors + 1;
    end
end
cer = char_errors / length(test_text) * 100;

fprintf('\nPerformance:\n');
fprintf('Bit Error Rate: %.2f%% (%d/%d bits)\n', ber, bit_errors, length(message_bits));
fprintf('Character Error Rate: %.2f%% (%d/%d chars)\n', cer, char_errors, length(test_text));
fprintf('Frame Success Rate: %.1f%% (%d/%d frames)\n', valid_frames/num_frames*100, valid_frames, num_frames);

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
plot(real(corrected_signal), 'b', 'LineWidth', 1);
hold on;
plot(imag(corrected_signal), 'r', 'LineWidth', 1);
title('Frequency Corrected Signal');
xlabel('Sample');
ylabel('Amplitude');
legend('Real', 'Imaginary');
grid on;

subplot(2,2,3);
data_constellation = rx_data(1:samples_per_symbol:end);
scatter(real(data_constellation), imag(data_constellation), 'bo');
hold on;
scatter(real(qpsk_map), imag(qpsk_map), 'rx', 'LineWidth', 2, 'SizeData', 100);
title('QPSK Constellation');
xlabel('In-phase');
ylabel('Quadrature');
legend('Received', 'Ideal');
grid on;
axis equal;

subplot(2,2,4);
% frame success rate vs frame number
frame_success = zeros(1, num_frames);
for i = 1:num_frames
    start_idx = (i-1) * (frame_size + 16) + 1;
    end_idx = min(i * (frame_size + 16), length(decoded_frames));
    if end_idx - start_idx + 1 >= frame_size + 16
        frame_data = decoded_frames(start_idx:end_idx);
        frame_success(i) = ~crc16_check(frame_data);
    end
end
bar(1:num_frames, frame_success);
title('Frame Success (CRC Pass)');
xlabel('Frame Number');
ylabel('Success (1=Pass, 0=Fail)');
grid on;

%% save results
save('exercise_4_2_3_results.mat', 'test_text', 'recovered_text', 'ber', 'cer', ...
     'hamming_corrections', 'valid_frames', 'num_frames');

saveas(gcf, 'exercise_4_2_3_frame_decoding.png');

fprintf('\nExercise 4.2.3 completed successfully!\n');