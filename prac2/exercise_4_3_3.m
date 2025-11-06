%% COMS4105 Practical 2 - Exercise 4.3.3
% concatenated rs + convolutional coding with 16-qam
% advanced error correction for frame type 0011

clear; close all; clc;

%% coding parameters
% outer code: reed-solomon (255,239)
n_rs = 255; k_rs = 239; t_rs = 8;

% inner code: convolutional rate 1/3
constraint_length = 3;
num_states = 4;

fprintf('=== Concatenated RS + Convolutional Coding (16-QAM) ===\n');
fprintf('Outer: RS(%d,%d) over GF(256)\n', n_rs, k_rs);
fprintf('Inner: Rate 1/3 convolutional, K=%d\n', constraint_length);
fprintf('Modulation: 16-QAM\n');

%% signal parameters
fs = 2e6;
samples_per_symbol = 16;

% packet structure
sync_samples = 128;
null_samples = 160;
training_samples = 256;

fprintf('Sample rate: %.1f MS/s\n', fs/1e6);

%% concatenated coding functions
function encoded = rs_encode_simple(message)
    % simplified rs encoder for demonstration
    k = 239; n = 255;
    
    % pad to k symbols
    if length(message) < k
        message = [message zeros(1, k - length(message))];
    end
    
    % add 16 parity symbols (simplified)
    parity = zeros(1, 16);
    for i = 1:16
        parity(i) = mod(sum(message) + i, 256);
    end
    
    encoded = [message parity];
end

function [decoded, errors_corrected] = rs_decode_simple(received)
    % simplified rs decoder
    k = 239; t = 8;
    
    % extract message part
    message = received(1:k);
    parity = received(k+1:end);
    
    % simplified syndrome calculation
    errors_corrected = 0;
    syndrome = zeros(1, 16);
    
    for i = 1:16
        expected_parity = mod(sum(message) + i, 256);
        syndrome(i) = mod(parity(i) - expected_parity, 256);
    end
    
    % simple error correction (up to t errors)
    error_positions = find(syndrome ~= 0);
    if length(error_positions) <= t && ~isempty(error_positions)
        for pos = error_positions(1:min(length(error_positions), t))
            if pos <= length(message)
                message(pos) = mod(message(pos) + syndrome(pos), 256);
                errors_corrected = errors_corrected + 1;
            end
        end
    end
    
    decoded = message;
end

function encoded = conv_encode(data_bits)
    % convolutional encoder rate 1/3
    state = 1;
    encoded = [];
    next_state = [1 3; 2 4; 1 3; 2 4];
    outputs = [0 7; 3 4; 1 6; 2 5];
    
    for bit = data_bits
        output_decimal = outputs(state, bit+1);
        output_bits = de2bi(output_decimal, 3, 'left-msb');
        encoded = [encoded output_bits];
        state = next_state(state, bit+1);
    end
end

function decoded = viterbi_decode(received_bits)
    % viterbi decoder for rate 1/3 convolutional
    num_bits = length(received_bits) / 3;
    num_states = 4;
    
    % state tables
    next_state = [1 3; 2 4; 1 3; 2 4];
    outputs = [0 7; 3 4; 1 6; 2 5];
    
    % initialize metrics
    path_metrics = inf(num_states, 1);
    path_metrics(1) = 0;
    survivor_paths = cell(num_states, 1);
    for i = 1:num_states
        survivor_paths{i} = [];
    end
    
    % process each symbol
    for t = 1:num_bits
        rx_symbol = received_bits((t-1)*3+1:t*3);
        new_metrics = inf(num_states, 1);
        new_paths = cell(num_states, 1);
        
        for curr_state = 1:num_states
            if isinf(path_metrics(curr_state))
                continue;
            end
            
            for input_bit = 0:1
                next_st = next_state(curr_state, input_bit+1);
                expected_output = de2bi(outputs(curr_state, input_bit+1), 3, 'left-msb');
                
                distance = sum(rx_symbol ~= expected_output);
                new_metric = path_metrics(curr_state) + distance;
                
                if new_metric < new_metrics(next_st)
                    new_metrics(next_st) = new_metric;
                    new_paths{next_st} = [survivor_paths{curr_state} input_bit];
                end
            end
        end
        
        path_metrics = new_metrics;
        survivor_paths = new_paths;
    end
    
    [~, best_state] = min(path_metrics);
    decoded = survivor_paths{best_state};
end

%% create test message
test_message = 'CONCATENATED CODING: RS + CONVOLUTIONAL WITH 16QAM FOR FRAME 0011';
fprintf('\nOriginal message: "%s"\n', test_message);

% convert to 8-bit symbols for rs encoding
ascii_vals = double(test_message);
message_symbols = ascii_vals;

% rs outer encoding
symbols_per_block = k_rs - 4; % reserve for header
num_rs_blocks = ceil(length(message_symbols) / symbols_per_block);

rs_encoded = [];
for block_idx = 1:num_rs_blocks
    start_idx = (block_idx - 1) * symbols_per_block + 1;
    end_idx = min(block_idx * symbols_per_block, length(message_symbols));
    
    block_data = message_symbols(start_idx:end_idx);
    
    % pad if needed
    if length(block_data) < symbols_per_block
        block_data = [block_data zeros(1, symbols_per_block - length(block_data))];
    end
    
    % add header and encode
    full_block = [length(message_symbols) block_idx num_rs_blocks block_idx block_data];
    if length(full_block) < k_rs
        full_block = [full_block zeros(1, k_rs - length(full_block))];
    end
    
    rs_codeword = rs_encode_simple(full_block);
    rs_encoded = [rs_encoded rs_codeword];
end

% convert rs symbols to bits
rs_bits = [];
for symbol = rs_encoded
    symbol_bits = de2bi(symbol, 8, 'left-msb');
    rs_bits = [rs_bits symbol_bits];
end

% add termination bits for convolutional
terminated_bits = [rs_bits 0 0];

% convolutional inner encoding
conv_encoded = conv_encode(terminated_bits);

fprintf('Message: %d symbols -> RS: %d symbols -> Bits: %d -> Conv: %d\n', ...
    length(message_symbols), length(rs_encoded), length(rs_bits), length(conv_encoded));

%% create packet structure
% sync: 10101011 (bpsk)
sync_bits = [1 0 1 0 1 0 1 1];

% training: 11110011 10100000 (16-qam, repeated)
training_bits = [1 1 1 1 0 0 1 1 1 0 1 0 0 0 0 0];
training_repeated = [training_bits training_bits];

% frame type: 0011 (16-qam)
frame_type_bits = [0 0 1 1];

% complete packet
packet_bits = [sync_bits zeros(1,16) training_repeated frame_type_bits conv_encoded];

fprintf('Frame type: 0011\n');
fprintf('Total packet: %d bits\n', length(packet_bits));

%% 16-qam modulation
% 16-qam constellation (gray mapping)
qam16_map = [3+3j, 3+1j, 3-3j, 3-1j, 1+3j, 1+1j, 1-3j, 1-1j, ...
            -3+3j, -3+1j, -3-3j, -3-1j, -1+3j, -1+1j, -1-3j, -1-1j]/sqrt(10);

% sync: bpsk
sync_symbols = 2 * sync_bits - 1;

% training: 16-qam
training_symbols = [];
for i = 1:4:length(training_repeated)
    if i+3 <= length(training_repeated)
        bit_quad = training_repeated(i:i+3);
        idx = bi2de(bit_quad, 'left-msb') + 1;
        training_symbols = [training_symbols qam16_map(idx)];
    end
end

% frame type: 16-qam
frame_symbols = [];
for i = 1:4:length(frame_type_bits)
    bit_quad = frame_type_bits(i:i+3);
    idx = bi2de(bit_quad, 'left-msb') + 1;
    frame_symbols = [frame_symbols qam16_map(idx)];
end

% data: 16-qam
data_symbols = [];
for i = 1:4:length(conv_encoded)
    if i+3 <= length(conv_encoded)
        bit_quad = conv_encoded(i:i+3);
        idx = bi2de(bit_quad, 'left-msb') + 1;
        data_symbols = [data_symbols qam16_map(idx)];
    end
end

%% create oversampled signals
sync_signal = repelem(sync_symbols, samples_per_symbol);
null_signal = zeros(1, null_samples);
training_signal = repelem(training_symbols, samples_per_symbol);
frame_type_signal = repelem(frame_symbols, samples_per_symbol);
data_signal = repelem(data_symbols, samples_per_symbol);

complete_signal = [sync_signal null_signal training_signal frame_type_signal data_signal];

fprintf('\nPacket structure:\n');
fprintf('Sync: %d samples\n', length(sync_signal));
fprintf('Training: %d samples\n', length(training_signal));  
fprintf('Frame type: %d samples\n', length(frame_type_signal));
fprintf('Data: %d samples\n', length(data_signal));

%% channel simulation
snr_db = 4;  % low snr to test concatenated coding
freq_offset_hz = 2000;

% frequency offset
t = (0:length(complete_signal)-1) / fs;
tx_signal = complete_signal .* exp(1j * 2 * pi * freq_offset_hz * t);

% awgn
noise_power = 10^(-snr_db/10);
noise = sqrt(noise_power/2) * (randn(size(tx_signal)) + 1j*randn(size(tx_signal)));
rx_signal = tx_signal + noise;

fprintf('\nChannel: SNR = %d dB, Freq offset = %d Hz\n', snr_db, freq_offset_hz);

%% synchronization and frame detection
sync_template = repelem(2 * sync_bits - 1, samples_per_symbol);
correlation = abs(xcorr(rx_signal, sync_template));
[~, peak_idx] = max(correlation);
sync_start = peak_idx - length(sync_template) - length(rx_signal) + 2;

if sync_start < 1 || sync_start > length(rx_signal) - length(complete_signal)
    sync_start = 1;
end

rx_packet = rx_signal(sync_start:sync_start + length(complete_signal) - 1);

% frequency synchronization using training sequence
training_start = length(sync_signal) + length(null_signal) + 1;
training_end = training_start + length(training_signal) - 1;

rx_training = rx_packet(training_start:training_end);
known_training = repelem(training_symbols, samples_per_symbol);

% estimate frequency offset
phase_diff = angle(sum(conj(known_training) .* rx_training));
estimated_offset = phase_diff / (2 * pi * length(training_signal) / fs);

fprintf('Estimated frequency offset: %.0f Hz\n', estimated_offset);

% apply correction
t_packet = (0:length(rx_packet)-1) / fs;
correction = exp(-1j * 2 * pi * estimated_offset * t_packet);
corrected_packet = rx_packet .* correction;

%% frame type detection
frame_type_start = length(sync_signal) + length(null_signal) + length(training_signal) + 1;
frame_type_end = frame_type_start + length(frame_type_signal) - 1;

rx_frame_type = corrected_packet(frame_type_start:frame_type_end);

% 16-qam demodulation for frame type
frame_type_demod = [];
for i = 1:samples_per_symbol:length(rx_frame_type)
    if i + samples_per_symbol - 1 <= length(rx_frame_type)
        symbol = mean(rx_frame_type(i:i+samples_per_symbol-1));
        
        % find closest constellation point
        [~, idx] = min(abs(qam16_map - symbol));
        bits = de2bi(idx-1, 4, 'left-msb');
        frame_type_demod = [frame_type_demod bits];
    end
end

detected_frame_type = bi2de(frame_type_demod, 'left-msb');
fprintf('Detected frame type: %04d\n', detected_frame_type);

%% data demodulation
data_start = frame_type_end + 1;
data_end = min(data_start + length(data_signal) - 1, length(corrected_packet));
rx_data = corrected_packet(data_start:data_end);

% 16-qam demodulation
demod_bits = [];
for i = 1:samples_per_symbol:length(rx_data)
    if i + samples_per_symbol - 1 <= length(rx_data)
        symbol = mean(rx_data(i:i+samples_per_symbol-1));
        
        % find closest constellation point
        [~, idx] = min(abs(qam16_map - symbol));
        bits = de2bi(idx-1, 4, 'left-msb');
        demod_bits = [demod_bits bits];
    end
end

% ensure proper length for viterbi
target_length = floor(length(demod_bits) / 3) * 3;
demod_bits = demod_bits(1:target_length);

fprintf('Demodulated %d bits for concatenated decoding\n', length(demod_bits));

%% inner convolutional decoding
conv_decoded = viterbi_decode(demod_bits);

% remove termination bits
if length(conv_decoded) >= 2
    conv_decoded = conv_decoded(1:end-2);
end

% convert to symbols for rs decoding
rs_rx_symbols = [];
for i = 1:8:length(conv_decoded)
    if i+7 <= length(conv_decoded)
        symbol_bits = conv_decoded(i:i+7);
        symbol_val = bi2de(symbol_bits, 'left-msb');
        rs_rx_symbols = [rs_rx_symbols symbol_val];
    end
end

fprintf('Convolutional decoded: %d bits -> %d symbols\n', length(conv_decoded), length(rs_rx_symbols));

%% outer reed-solomon decoding
decoded_message = [];
total_rs_corrections = 0;

for block_idx = 1:num_rs_blocks
    start_idx = (block_idx - 1) * n_rs + 1;
    end_idx = min(block_idx * n_rs, length(rs_rx_symbols));
    
    if end_idx - start_idx + 1 >= n_rs
        rx_block = rs_rx_symbols(start_idx:end_idx);
        
        [rs_decoded_block, corrections] = rs_decode_simple(rx_block);
        total_rs_corrections = total_rs_corrections + corrections;
        
        % extract message (skip header)
        block_message = rs_decoded_block(5:end);
        
        % handle last block length
        if block_idx == num_rs_blocks
            remaining = length(message_symbols) - (block_idx-1) * symbols_per_block;
            if remaining > 0 && remaining <= length(block_message)
                block_message = block_message(1:remaining);
            end
        else
            block_message = block_message(1:symbols_per_block);
        end
        
        decoded_message = [decoded_message block_message];
    end
end

% truncate to original length
decoded_message = decoded_message(1:min(length(decoded_message), length(message_symbols)));

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
fprintf('RS corrections: %d symbol errors\n', total_rs_corrections);
fprintf('Code rate: %.3f (RS) × %.3f (Conv) = %.3f overall\n', k_rs/n_rs, 1/3, (k_rs/n_rs)/3);

%% plot results
figure(1);
subplot(2,3,1);
plot(real(complete_signal), 'b', 'LineWidth', 1);
hold on;
plot(imag(complete_signal), 'r', 'LineWidth', 1);
title('Transmitted Signal');
xlabel('Sample');
ylabel('Amplitude');
legend('Real', 'Imaginary');
grid on;

subplot(2,3,2);
plot(real(corrected_packet), 'b', 'LineWidth', 1);
hold on;
plot(imag(corrected_packet), 'r', 'LineWidth', 1);
title('Frequency Corrected Signal');
xlabel('Sample');
ylabel('Amplitude');
legend('Real', 'Imaginary');
grid on;

subplot(2,3,3);
data_constellation = rx_data(1:samples_per_symbol:end);
scatter(real(data_constellation), imag(data_constellation), 'bo');
hold on;
scatter(real(qam16_map), imag(qam16_map), 'rx', 'LineWidth', 2, 'SizeData', 100);
title('16-QAM Constellation');
xlabel('In-phase');
ylabel('Quadrature');
legend('Received', 'Ideal');
grid on;
axis equal;

subplot(2,3,4);
% show coding stages
original_bits = length(rs_bits);
conv_bits = length(conv_encoded);
demod_errors = sum(conv_encoded(1:min(end,length(demod_bits))) ~= demod_bits(1:min(end,length(conv_encoded))));
conv_errors = sum(rs_bits(1:min(end,length(conv_decoded))) ~= conv_decoded(1:min(end,length(rs_bits))));

stage_data = [demod_errors, conv_errors, symbol_errors];
bar({'After 16-QAM', 'After Viterbi', 'After RS'}, stage_data);
title('Concatenated Error Correction');
ylabel('Number of Errors');
grid on;

subplot(2,3,5);
% ber vs snr simulation (simplified)
snr_range = 0:8;
ber_uncoded = qfunc(sqrt(2*10.^(snr_range/10)));
ber_coded = ber_uncoded.^3 * (k_rs/n_rs)/3; % approximation
semilogy(snr_range, ber_uncoded, 'r-', 'LineWidth', 2);
hold on;
semilogy(snr_range, ber_coded, 'b-', 'LineWidth', 2);
plot(snr_db, max(ser/100, 1e-6), 'bo', 'MarkerSize', 8, 'LineWidth', 2);
title('Coding Gain');
xlabel('SNR (dB)');
ylabel('Error Rate');
legend('Uncoded', 'RS+Conv', 'Measured');
grid on;

subplot(2,3,6);
% code rate and efficiency
rates = [1, k_rs/n_rs, 1/3, (k_rs/n_rs)/3];
rate_labels = {'Uncoded', 'RS Only', 'Conv Only', 'Concatenated'};
bar(rates);
set(gca, 'XTickLabel', rate_labels);
title('Code Rates');
ylabel('Rate');
grid on;

%% save results
save('exercise_4_3_3_results.mat', 'test_message', 'recovered_text', 'ser', 'cer', ...
     'detected_frame_type', 'estimated_offset', 'total_rs_corrections');

saveas(gcf, 'exercise_4_3_3_concatenated.png');

fprintf('\nExercise 4.3.3 completed successfully!\n');