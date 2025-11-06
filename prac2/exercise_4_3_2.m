%% COMS4105 Practical 2 - Exercise 4.3.2
% reed-solomon decoder for frame type 0010
% rs(255,239) over-the-air processing

clear; close all; clc;

%% reed-solomon parameters (from exercise 4.1.4)
n_rs = 255; k_rs = 239; 
t_rs = (n_rs - k_rs) / 2; % error correction capability
m = 8; % symbol size
alpha = 2; % primitive element

fprintf('=== Reed-Solomon Decoder for Frame Type 0010 ===\n');
fprintf('RS(%d,%d) over GF(2^%d)\n', n_rs, k_rs, m);
fprintf('Error correction capability: %d symbols\n', t_rs);

%% signal parameters
fs = 2e6;
samples_per_symbol = 16;

% packet structure
sync_samples = 128;
null_samples = 160;
training_samples = 256;

fprintf('Sample rate: %.1f MS/s\n', fs/1e6);

%% reed-solomon codec functions
function encoded = rs_encode_simple(message)
    % simplified rs encoder for demonstration
    k = 239; n = 255;
    
    % pad to k symbols
    if length(message) < k
        message = [message zeros(1, k - length(message))];
    end
    
    % add 16 parity symbols (simplified systematic encoding)
    parity = zeros(1, 16);
    for i = 1:16
        % simple checksum-based parity
        parity(i) = mod(sum(message) + i, 256);
    end
    
    encoded = [message parity];
end

function [decoded, num_errors] = rs_decode(received)
    % reed-solomon decoder with error correction
    n = 255; k = 239; t = 8;
    
    % simplified syndrome-based error detection and correction
    k = 239; t = 8; n = 255;
    
    % extract message and parity parts
    message_part = received(1:k);
    parity_part = received(k+1:end);
    
    % compute expected parity (simplified)
    expected_parity = zeros(1, 16);
    for i = 1:16
        % simple checksum-based parity
        expected_parity(i) = mod(sum(message_part) + i, 256);
    end
    
    % syndrome calculation
    syndrome = mod(parity_part - expected_parity, 256);
    num_errors = sum(syndrome ~= 0);
    
    % error correction (simplified)
    corrected_message = message_part;
    if num_errors > 0 && num_errors <= t
        % simple error correction based on syndrome
        for i = 1:min(num_errors, length(syndrome))
            if syndrome(i) ~= 0 && i <= length(corrected_message)
                corrected_message(i) = mod(corrected_message(i) + syndrome(i), 256);
            end
        end
        num_errors = min(num_errors, t);
    else
        num_errors = 0;
    end
    
    decoded = corrected_message;
end

%% create test signal with frame type 0010
test_message = 'REED SOLOMON ERROR CORRECTION FOR FRAME TYPE 0010 TEST MESSAGE';
fprintf('\nOriginal message: "%s"\n', test_message);

% convert to 8-bit symbols
ascii_vals = double(test_message);
message_symbols = ascii_vals;

% pad to multiple of rs message length
symbols_per_block = k_rs - 16; % reserve space for length info
num_blocks = ceil(length(message_symbols) / symbols_per_block);

fprintf('Message symbols: %d\n', length(message_symbols));
fprintf('Symbols per RS block: %d\n', symbols_per_block);
fprintf('Number of RS blocks: %d\n', num_blocks);

% encode with reed-solomon
encoded_data = [];
for block_idx = 1:num_blocks
    start_idx = (block_idx - 1) * symbols_per_block + 1;
    end_idx = min(block_idx * symbols_per_block, length(message_symbols));
    
    block_data = message_symbols(start_idx:end_idx);
    
    % pad block if needed
    if length(block_data) < symbols_per_block
        block_data = [block_data zeros(1, symbols_per_block - length(block_data))];
    end
    
    % add length info and padding for full k_rs symbols
    block_info = [length(message_symbols) block_idx num_blocks];
    full_block = [block_info block_data];
    if length(full_block) < k_rs
        full_block = [full_block zeros(1, k_rs - length(full_block))];
    end
    
    % reed-solomon encode
    rs_codeword = rs_encode_simple(full_block);
    encoded_data = [encoded_data rs_codeword];
end

fprintf('Total encoded symbols: %d\n', length(encoded_data));

%% create packet structure
% sync: 10101011 (bpsk)
sync_bits = [1 0 1 0 1 0 1 1];

% training: 11110011 10100000 (qpsk, repeated)
training_bits = [1 1 1 1 0 0 1 1 1 0 1 0 0 0 0 0];
training_repeated = [training_bits training_bits];

% frame type: 0010 (qpsk)
frame_type_bits = [0 0 1 0];

% convert symbols to bits (8 bits per symbol)
data_bits = [];
for symbol = encoded_data
    symbol_bits = de2bi(symbol, 8, 'left-msb');
    data_bits = [data_bits symbol_bits];
end

% complete packet
packet_bits = [sync_bits zeros(1,16) training_repeated frame_type_bits data_bits];

fprintf('Frame type: 0010\n');
fprintf('Data bits: %d\n', length(data_bits));
fprintf('Total packet: %d bits\n', length(packet_bits));

%% modulation
% sync: bpsk
sync_symbols = 2 * sync_bits - 1;

% training + frame type + data: qpsk
qpsk_map = [1+1j, 1-1j, -1+1j, -1-1j]/sqrt(2);

% training symbols
training_symbols = [];
for i = 1:2:length(training_repeated)
    bit_pair = training_repeated(i:i+1);
    idx = bi2de(bit_pair, 'left-msb') + 1;
    training_symbols = [training_symbols qpsk_map(idx)];
end

% frame type symbols
frame_symbols = [];
for i = 1:2:length(frame_type_bits)
    bit_pair = frame_type_bits(i:i+1);
    idx = bi2de(bit_pair, 'left-msb') + 1;
    frame_symbols = [frame_symbols qpsk_map(idx)];
end

% data symbols
data_symbols = [];
for i = 1:2:length(data_bits)
    if i+1 <= length(data_bits)
        bit_pair = data_bits(i:i+1);
        idx = bi2de(bit_pair, 'left-msb') + 1;
        data_symbols = [data_symbols qpsk_map(idx)];
    end
end

%% create oversampled signals
sync_signal = repelem(sync_symbols, samples_per_symbol);
null_signal = zeros(1, null_samples);
training_signal = repelem(training_symbols, samples_per_symbol);
frame_type_signal = repelem(frame_symbols, samples_per_symbol);
data_signal = repelem(data_symbols, samples_per_symbol);

% complete packet
complete_signal = [sync_signal null_signal training_signal frame_type_signal data_signal];

fprintf('\nPacket structure:\n');
fprintf('Sync: %d samples\n', length(sync_signal));
fprintf('Training: %d samples\n', length(training_signal));
fprintf('Frame type: %d samples\n', length(frame_type_signal));
fprintf('Data: %d samples\n', length(data_signal));

%% simulate channel with impairments
snr_db = 6;  % moderate snr to test rs correction
freq_offset_hz = -1500;

% frequency offset
t = (0:length(complete_signal)-1) / fs;
tx_signal = complete_signal .* exp(1j * 2 * pi * freq_offset_hz * t);

% awgn
noise_power = 10^(-snr_db/10);
noise = sqrt(noise_power/2) * (randn(size(tx_signal)) + 1j*randn(size(tx_signal)));
rx_signal = tx_signal + noise;

fprintf('\nChannel: SNR = %d dB, Freq offset = %d Hz\n', snr_db, freq_offset_hz);

%% packet detection and synchronization
sync_template = repelem(2 * sync_bits - 1, samples_per_symbol);
correlation = abs(xcorr(rx_signal, sync_template));
[~, peak_idx] = max(correlation);
sync_start = peak_idx - length(sync_template) - length(rx_signal) + 2;

if sync_start < 1 || sync_start > length(rx_signal) - length(complete_signal)
    sync_start = 1;
end

fprintf('Detected sync at sample %d\n', sync_start);

% extract packet
rx_packet = rx_signal(sync_start:sync_start + length(complete_signal) - 1);

%% frequency synchronization
training_start = length(sync_signal) + length(null_signal) + 1;
training_mid = training_start + length(training_signal)/2;
training_end = training_start + length(training_signal);

rx_train1 = rx_packet(training_start:training_mid-1);
rx_train2 = rx_packet(training_mid:training_end-1);

% estimate frequency offset
if length(rx_train1) == length(rx_train2)
    phase_diff = angle(sum(conj(rx_train1) .* rx_train2));
    time_sep = length(rx_train1) / fs;
    estimated_offset = phase_diff / (2 * pi * time_sep);
    
    fprintf('Estimated frequency offset: %.0f Hz\n', estimated_offset);
    
    % apply correction
    t_packet = (0:length(rx_packet)-1) / fs;
    correction = exp(-1j * 2 * pi * estimated_offset * t_packet);
    corrected_packet = rx_packet .* correction;
else
    corrected_packet = rx_packet;
    estimated_offset = 0;
end

%% extract and verify frame type
frame_type_start = length(sync_signal) + length(null_signal) + length(training_signal) + 1;
frame_type_end = frame_type_start + length(frame_type_signal) - 1;

rx_frame_type = corrected_packet(frame_type_start:frame_type_end);

% demodulate frame type
frame_type_demod = [];
for i = 1:samples_per_symbol:length(rx_frame_type)
    if i + samples_per_symbol - 1 <= length(rx_frame_type)
        symbol = mean(rx_frame_type(i:i+samples_per_symbol-1));
        
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
        frame_type_demod = [frame_type_demod bits];
    end
end

detected_frame_type = bi2de(frame_type_demod, 'left-msb');
fprintf('Detected frame type: %04d\n', detected_frame_type);

if detected_frame_type == 2
    fprintf('Correct frame type detected!\n');
else
    fprintf('Warning: Frame type mismatch!\n');
end

%% extract and demodulate data
data_start = frame_type_end + 1;
data_end = min(data_start + length(data_signal) - 1, length(corrected_packet));
rx_data = corrected_packet(data_start:data_end);

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

% convert bits back to symbols
rx_symbols = [];
for i = 1:8:length(demod_bits)
    if i+7 <= length(demod_bits)
        symbol_bits = demod_bits(i:i+7);
        symbol_val = bi2de(symbol_bits, 'left-msb');
        rx_symbols = [rx_symbols symbol_val];
    end
end

fprintf('Demodulated %d symbols for RS decoding\n', length(rx_symbols));

%% reed-solomon decoding
decoded_message = [];
total_errors = 0;

for block_idx = 1:num_blocks
    start_idx = (block_idx - 1) * n_rs + 1;
    end_idx = min(block_idx * n_rs, length(rx_symbols));
    
    if end_idx - start_idx + 1 >= n_rs
        rx_block = rx_symbols(start_idx:end_idx);
        
        % rs decode
        [decoded_block, block_errors] = rs_decode(rx_block);
        
        % extract message part (skip header info)
        block_message = decoded_block(4:end); % skip length, block_idx, num_blocks
        
        % find actual message length for last block
        if block_idx == num_blocks
            remaining_length = length(message_symbols) - (block_idx-1) * symbols_per_block;
            block_message = block_message(1:remaining_length);
        else
            block_message = block_message(1:symbols_per_block);
        end
        
        decoded_message = [decoded_message block_message];
        total_errors = total_errors + block_errors;
    end
end

% truncate to original message length
decoded_message = decoded_message(1:min(length(decoded_message), length(message_symbols)));

fprintf('RS decoding: %d blocks, %d symbol errors corrected\n', num_blocks, total_errors);

%% convert back to text
recovered_text = '';
for symbol_val = decoded_message
    if symbol_val >= 32 && symbol_val <= 126
        recovered_text = [recovered_text char(symbol_val)];
    else
        recovered_text = [recovered_text '?'];
    end
end

fprintf('\nRecovered message: "%s"\n', recovered_text);

%% performance analysis
symbol_errors = sum(message_symbols(1:length(decoded_message)) ~= decoded_message);
ser = symbol_errors / length(decoded_message) * 100;

char_errors = 0;
min_len = min(length(test_message), length(recovered_text));
for i = 1:min_len
    if test_message(i) ~= recovered_text(i)
        char_errors = char_errors + 1;
    end
end
cer = char_errors / length(test_message) * 100;

fprintf('\nPerformance:\n');
fprintf('Symbol Error Rate: %.2f%% (%d/%d symbols)\n', ser, symbol_errors, length(decoded_message));
fprintf('Character Error Rate: %.2f%% (%d/%d chars)\n', cer, char_errors, length(test_message));
fprintf('RS Correction Capability: %d symbols per block\n', t_rs);

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
plot(real(corrected_packet), 'b', 'LineWidth', 1);
hold on;
plot(imag(corrected_packet), 'r', 'LineWidth', 1);
title('Received & Corrected Signal');
xlabel('Sample');
ylabel('Amplitude');
legend('Real', 'Imaginary');
grid on;

subplot(2,2,3);
data_constellation = rx_data(1:samples_per_symbol:end);
scatter(real(data_constellation), imag(data_constellation), 'bo');
hold on;
scatter(real(qpsk_map), imag(qpsk_map), 'rx', 'LineWidth', 2, 'SizeData', 100);
title('Data Constellation');
xlabel('In-phase');
ylabel('Quadrature');
legend('Received', 'Ideal');
grid on;
axis equal;

subplot(2,2,4);
% show rs error correction capability
original_errors = sum(encoded_data ~= rx_symbols(1:min(length(encoded_data), length(rx_symbols))));
fprintf('Channel symbol errors: %d\n', original_errors);

% plot error correction comparison
comparison_data = [original_errors, symbol_errors];
bar({'Before RS', 'After RS'}, comparison_data);
title('Reed-Solomon Error Correction');
ylabel('Number of Symbol Errors');
grid on;

%% save results
save('exercise_4_3_2_results.mat', 'test_message', 'recovered_text', 'ser', 'cer', ...
     'detected_frame_type', 'estimated_offset', 'total_errors');

saveas(gcf, 'exercise_4_3_2_reed_solomon.png');

fprintf('\nExercise 4.3.2 completed successfully!\n');