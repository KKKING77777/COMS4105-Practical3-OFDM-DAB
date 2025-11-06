%% COMS4105 Practical 2 - Exercise 4.3.1
% convolutional decoder for frame type 0001
% over-the-air frame processing

clear; close all; clc;

%% convolutional code parameters (from preparation)
% generator polynomials: x0=1+p, x1=1, x2=1+p+p^2
constraint_length = 3;
num_states = 4;

% state transition tables
next_state = [1 3; 2 4; 1 3; 2 4];  % [current_state][input] -> next_state
outputs = [0 7; 3 4; 1 6; 2 5];     % [current_state][input] -> 3-bit output

fprintf('=== Convolutional Decoder for Frame Type 0001 ===\n');
fprintf('Code rate: 1/3, Constraint length: %d\n', constraint_length);

%% signal parameters
fs = 2e6;
samples_per_symbol = 16;

% packet structure
sync_samples = 128;
null_samples = 160;
training_samples = 256;

fprintf('Sample rate: %.1f MS/s\n', fs/1e6);

%% viterbi decoder function
function decoded = viterbi_decode(received_bits)
    num_bits = length(received_bits) / 3;
    num_states = 4;
    
    % state tables
    next_state = [1 3; 2 4; 1 3; 2 4];
    outputs = [0 7; 3 4; 1 6; 2 5];
    
    % initialize
    path_metrics = inf(num_states, 1);
    path_metrics(1) = 0;  % start from state 1 (00)
    survivor_paths = cell(num_states, 1);
    for i = 1:num_states
        survivor_paths{i} = [];
    end
    
    % process each 3-bit symbol
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
                
                % hamming distance
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
    
    % find best path
    [~, best_state] = min(path_metrics);
    decoded = survivor_paths{best_state};
end

%% create test signal with frame type 0001
test_message = 'CONVOLUTIONAL CODE TEST MESSAGE FOR FRAME 0001';
fprintf('\nOriginal message: "%s"\n', test_message);

% convert to bits
ascii_vals = double(test_message);
message_bits = [];
for val = ascii_vals
    char_bits = de2bi(val, 8, 'left-msb');
    message_bits = [message_bits char_bits];
end

% add termination bits
terminated_bits = [message_bits 0 0];

% convolutional encoding
function encoded = conv_encode(data_bits)
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

encoded_message = conv_encode(terminated_bits);
fprintf('Message bits: %d\n', length(message_bits));
fprintf('Encoded bits: %d (rate = %.3f)\n', length(encoded_message), length(terminated_bits)/length(encoded_message));

%% create packet structure
% sync: 10101011 (bpsk)
sync_bits = [1 0 1 0 1 0 1 1];

% training: 11110011 10100000 (qpsk, repeated)
training_bits = [1 1 1 1 0 0 1 1 1 0 1 0 0 0 0 0];
training_repeated = [training_bits training_bits];

% frame type: 0001 (qpsk)
frame_type_bits = [0 0 0 1];

% combine: sync + null + training + frame_type + encoded_message
packet_bits = [sync_bits zeros(1,16) training_repeated frame_type_bits encoded_message];

fprintf('Frame type: 0001\n');
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
for i = 1:2:length(encoded_message)
    if i+1 <= length(encoded_message)
        bit_pair = encoded_message(i:i+1);
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
snr_db = 8;  % lower snr to test error correction
freq_offset_hz = -800;

% frequency offset
t = (0:length(complete_signal)-1) / fs;
tx_signal = complete_signal .* exp(1j * 2 * pi * freq_offset_hz * t);

% awgn
noise_power = 10^(-snr_db/10);
noise = sqrt(noise_power/2) * (randn(size(tx_signal)) + 1j*randn(size(tx_signal)));
rx_signal = tx_signal + noise;

fprintf('\nChannel: SNR = %d dB, Freq offset = %d Hz\n', snr_db, freq_offset_hz);

%% packet detection and synchronization
% find sync pattern (simplified correlation)
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

if detected_frame_type == 1
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

% ensure proper length for viterbi (multiple of 3)
target_length = floor(length(demod_bits) / 3) * 3;
demod_bits = demod_bits(1:target_length);

fprintf('Demodulated %d bits for Viterbi decoding\n', length(demod_bits));

%% convolutional decoding with viterbi
decoded_bits = viterbi_decode(demod_bits);

% remove termination bits
if length(decoded_bits) >= 2
    decoded_message = decoded_bits(1:end-2);
else
    decoded_message = decoded_bits;
end

% match original message length
decoded_message = decoded_message(1:min(length(decoded_message), length(message_bits)));

fprintf('Decoded %d message bits\n', length(decoded_message));

%% convert back to text
recovered_text = '';
for i = 1:8:length(decoded_message)
    if i+7 <= length(decoded_message)
        char_bits = decoded_message(i:i+7);
        ascii_val = bi2de(char_bits, 'left-msb');
        if ascii_val >= 32 && ascii_val <= 126
            recovered_text = [recovered_text char(ascii_val)];
        else
            recovered_text = [recovered_text '?'];
        end
    end
end

fprintf('\nRecovered message: "%s"\n', recovered_text);

%% performance analysis
bit_errors = sum(message_bits(1:length(decoded_message)) ~= decoded_message);
ber = bit_errors / length(decoded_message) * 100;

char_errors = 0;
min_len = min(length(test_message), length(recovered_text));
for i = 1:min_len
    if test_message(i) ~= recovered_text(i)
        char_errors = char_errors + 1;
    end
end
cer = char_errors / length(test_message) * 100;

fprintf('\nPerformance:\n');
fprintf('Bit Error Rate: %.2f%% (%d/%d bits)\n', ber, bit_errors, length(decoded_message));
fprintf('Character Error Rate: %.2f%% (%d/%d chars)\n', cer, char_errors, length(test_message));

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
% show error correction capability
original_errors = sum(encoded_message ~= demod_bits(1:min(length(encoded_message), length(demod_bits))));
fprintf('Channel bit errors: %d\n', original_errors);

% plot bit error comparison
comparison_data = [original_errors, bit_errors];
bar({'Before Viterbi', 'After Viterbi'}, comparison_data);
title('Error Correction Performance');
ylabel('Number of Errors');
grid on;

%% save results
save('exercise_4_3_1_results.mat', 'test_message', 'recovered_text', 'ber', 'cer', ...
     'detected_frame_type', 'estimated_offset');

saveas(gcf, 'exercise_4_3_1_convolutional.png');

fprintf('\nExercise 4.3.1 completed successfully!\n');