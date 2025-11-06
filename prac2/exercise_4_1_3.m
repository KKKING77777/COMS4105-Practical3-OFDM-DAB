%% COMS4105 Practical 2 - Exercise 4.1.3
% convolutional encoder/decoder
% simplified conv code from preparation

clear; close all; clc;

%% convolutional code parameters (from preparation)
% generator polynomials:
% x0 = 1 + p
% x1 = 1  
% x2 = 1 + p + p^2
% constraint length k = 3 (2 memory elements)
% code rate = 1/3

constraint_length = 3;
code_rate = 1/3;
num_states = 2^(constraint_length-1); % 4 states

% generator polynomials in octal
gen_poly = [5 1 7]; % octal: 5=101, 1=001, 7=111

fprintf('=== Convolutional Code Parameters ===\n');
fprintf('Constraint length K = %d\n', constraint_length);
fprintf('Code rate = 1/%d\n', round(1/code_rate));
fprintf('Number of states = %d\n', num_states);
fprintf('Generator polynomials (octal): %s\n', mat2str(gen_poly));

%% state transition tables
% state transitions and outputs for each input bit
% states: 00, 01, 10, 11 (indexed as 1,2,3,4)

% next state table: [current_state][input_bit] -> next_state
next_state = [1 3;  % state 00: input 0->00, input 1->10
              2 4;  % state 01: input 0->01, input 1->11  
              1 3;  % state 10: input 0->00, input 1->10
              2 4]; % state 11: input 0->01, input 1->11

% output table: [current_state][input_bit] -> 3-bit output
outputs = [0 7;   % state 00: input 0->000, input 1->111
           3 4;   % state 01: input 0->011, input 1->100
           1 6;   % state 10: input 0->001, input 1->110  
           2 5];  % state 11: input 0->010, input 1->101

%% convolutional encoder
function encoded = conv_encode(data_bits)
    % encode using state machine
    state = 1; % start at state 00 (index 1)
    encoded = [];
    
    % state transition tables
    next_state = [1 3; 2 4; 1 3; 2 4];
    outputs = [0 7; 3 4; 1 6; 2 5];
    
    for bit = data_bits
        % get output for current state and input
        output_decimal = outputs(state, bit+1);
        output_bits = de2bi(output_decimal, 3, 'left-msb');
        encoded = [encoded output_bits];
        
        % update state
        state = next_state(state, bit+1);
    end
end

%% viterbi decoder  
function decoded = viterbi_decode(received_bits)
    num_bits = length(received_bits) / 3;
    num_states = 4;
    
    % state transition tables
    next_state = [1 3; 2 4; 1 3; 2 4];
    outputs = [0 7; 3 4; 1 6; 2 5];
    
    % initialize metrics and paths
    path_metrics = inf(num_states, 1);
    path_metrics(1) = 0; % start from state 1 (00)
    survivor_paths = cell(num_states, 1);
    
    for i = 1:num_states
        survivor_paths{i} = [];
    end
    
    % process each received symbol (3 bits)
    for t = 1:num_bits
        rx_symbol = received_bits((t-1)*3+1:t*3);
        new_metrics = inf(num_states, 1);
        new_paths = cell(num_states, 1);
        
        % for each current state
        for curr_state = 1:num_states
            if isinf(path_metrics(curr_state))
                continue;
            end
            
            % try both input bits (0 and 1)
            for input_bit = 0:1
                next_st = next_state(curr_state, input_bit+1);
                expected_output = de2bi(outputs(curr_state, input_bit+1), 3, 'left-msb');
                
                % calculate hamming distance
                distance = sum(rx_symbol ~= expected_output);
                new_metric = path_metrics(curr_state) + distance;
                
                % update if better path found
                if new_metric < new_metrics(next_st)
                    new_metrics(next_st) = new_metric;
                    new_paths{next_st} = [survivor_paths{curr_state} input_bit];
                end
            end
        end
        
        path_metrics = new_metrics;
        survivor_paths = new_paths;
    end
    
    % find best final state
    [~, best_state] = min(path_metrics);
    decoded = survivor_paths{best_state};
end

%% test with preparation example
fprintf('\n=== Preparation Example Test ===\n');
test_input = [1 0 1]; % from preparation question
encoded_test = conv_encode(test_input);
fprintf('Input bits: %s\n', mat2str(test_input));
fprintf('Encoded: %s\n', mat2str(encoded_test));

% add termination bits (pad with zeros)
padded_input = [test_input 0 0]; % pad to clear state
encoded_padded = conv_encode(padded_input);
fprintf('With termination: %s\n', mat2str(encoded_padded));

% test decoder with preparation received sequence
prep_received = [1 0 1 0 0 1 1 1 0 0 1 1 0 0 1]; % from question
fprintf('\nDecoding preparation sequence: %s\n', mat2str(prep_received));
decoded_prep = viterbi_decode(prep_received);
fprintf('Decoded bits: %s\n', mat2str(decoded_prep));

%% ber performance test
snr_db_range = 0:1:8;
num_bits_test = 1000; % bits per test

ber_uncoded = zeros(size(snr_db_range));
ber_conv = zeros(size(snr_db_range));

fprintf('\n=== BER Performance Test ===\n');

for snr_idx = 1:length(snr_db_range)
    snr_db = snr_db_range(snr_idx);
    snr_linear = 10^(snr_db/10);
    noise_var = 1 / (2 * snr_linear * code_rate); % account for code rate
    
    errors_uncoded = 0;
    errors_conv = 0;
    
    % test multiple blocks
    for block = 1:50
        % generate random data with termination
        data_bits = randi([0 1], 1, 20);
        terminated_bits = [data_bits 0 0]; % termination
        
        % convolutional encoding
        encoded_bits = conv_encode(terminated_bits);
        
        % bpsk modulation  
        tx_uncoded = 2 * data_bits - 1;
        tx_coded = 2 * encoded_bits - 1;
        
        % awgn channel
        noise_uncoded = sqrt(noise_var) * randn(1, length(data_bits));
        noise_coded = sqrt(noise_var) * randn(1, length(encoded_bits));
        
        rx_uncoded = tx_uncoded + noise_uncoded;
        rx_coded = tx_coded + noise_coded;
        
        % demodulation
        demod_uncoded = rx_uncoded > 0;
        demod_coded = rx_coded > 0;
        
        % convolutional decoding
        decoded_bits = viterbi_decode(demod_coded);
        decoded_data = decoded_bits(1:length(data_bits)); % remove termination
        
        % count errors
        errors_uncoded = errors_uncoded + sum(data_bits ~= demod_uncoded);
        errors_conv = errors_conv + sum(data_bits ~= decoded_data);
    end
    
    ber_uncoded(snr_idx) = errors_uncoded / (50 * length(data_bits));
    ber_conv(snr_idx) = errors_conv / (50 * length(data_bits));
    
    fprintf('SNR = %d dB: Uncoded BER = %.2e, Conv BER = %.2e\n', ...
            snr_db, ber_uncoded(snr_idx), ber_conv(snr_idx));
end

%% plot results
figure(1);
semilogy(snr_db_range, ber_uncoded, 'r-o', 'LineWidth', 2);
hold on;
semilogy(snr_db_range, ber_conv, 'b-s', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate');
title('BER Performance: Uncoded vs Convolutional (1/3 rate)');
legend('Uncoded BPSK', 'Convolutional Code', 'Location', 'southwest');

%% draw trellis diagram
figure(2);
states = {'00', '01', '10', '11'};
time_steps = 0:4;

% plot state nodes
for t = time_steps
    for s = 1:4
        plot(t, s, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'white');
        hold on;
        if t == 0
            text(t-0.3, s, states{s}, 'FontSize', 10);
        end
    end
end

% draw transitions (simplified for first few time steps)
transitions = {[1 1; 1 3], [2 2; 2 4], [3 1; 3 3], [4 2; 4 4]};
for s = 1:4
    for input = 0:1
        next_s = transitions{s}(input+1, 2);
        for t = 0:3
            if input == 0
                plot([t t+1], [s next_s], 'b-', 'LineWidth', 1);
            else
                plot([t t+1], [s next_s], 'r--', 'LineWidth', 1);
            end
        end
    end
end

xlabel('Time Step');
ylabel('State');
title('Convolutional Code Trellis Diagram');
ylim([0.5 4.5]);
grid on;
legend('Input 0', 'Input 1', 'Location', 'northeast');

%% save results
save('exercise_4_1_3_results.mat', 'snr_db_range', 'ber_uncoded', 'ber_conv');
saveas(figure(1), 'exercise_4_1_3_ber_performance.png');
saveas(figure(2), 'exercise_4_1_3_trellis.png');

fprintf('\nExercise 4.1.3 completed successfully!\n');