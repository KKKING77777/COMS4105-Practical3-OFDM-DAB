%% COMS4105 Practical 2 - Exercise 4.1.1 (CORRECTED)
% Hamming (15,11) encoder/decoder with BER testing
% 
% CORRECTIONS MADE:
% 1. Fixed H matrix construction - each column is binary representation of bit position
% 2. Corrected encoder to place information bits in non-power-of-2 positions
% 3. Fixed decoder syndrome interpretation - syndrome directly gives error position
% 4. Proper parity bit calculation using standard Hamming code rules
%
% KEY FEATURES:
% - Syndrome value directly indicates error position (1-indexed)
% - 100% single error correction capability
% - Systematic code structure for easy implementation
% - Proper coding gain demonstration at moderate to high SNR

clear; close all; clc;

%% hamming (15,11) parameters
n = 15; % codeword length
k = 11; % message length
t = 1;  % error correction capability

% Standard (15,11) Hamming code - systematic form
% For Hamming codes, parity check positions are at powers of 2: 1,2,4,8,...
% Information positions are at non-powers of 2: 3,5,6,7,9,10,11,12,13,14,15

% Construct H matrix directly for (15,11) Hamming code
% H matrix columns correspond to bit positions 1,2,3,...,15
% Each column is the binary representation of its position number
H = zeros(4, 15);
for i = 1:15
    H(:,i) = de2bi(i, 4, 'left-msb')';  % binary representation of position i
end

fprintf('Parity check matrix H:\n');
disp(H);

% For systematic form G = [I_k | P], extract P from H
% Information bit positions (non-powers of 2): [3,5,6,7,9,10,11,12,13,14,15]
info_positions = [3,5,6,7,9,10,11,12,13,14,15];
parity_positions = [1,2,4,8];  % positions 1,2,4,8

% Extract parity part P from H matrix
P = H(:, info_positions)';  % 11x4 matrix

% Identity matrix for information bits
I = eye(k);  % 11x11 identity matrix

% Systematic generator matrix G = [I_k | P]
G = [I P];  % 11x15 generator matrix

fprintf('=== Hamming (15,11) Code Parameters ===\n');
fprintf('Code length n = %d\n', n);
fprintf('Message length k = %d\n', k);
fprintf('Code rate = %.3f\n', k/n);
fprintf('Error correction capability t = %d\n', t);

% Display the structure
fprintf('Information bit positions: [');
fprintf('%d ', info_positions);
fprintf(']\n');
fprintf('Parity bit positions: [');
fprintf('%d ', parity_positions);
fprintf(']\n');

%% Hamming encoder function (corrected)
function codeword = hamming_encode(message, H, info_positions, parity_positions)
    % Encode k-bit message to n-bit codeword using standard Hamming (15,11)
    % 
    % ALGORITHM:
    % 1. Place information bits in non-power-of-2 positions (3,5,6,7,9,10,11,12,13,14,15)
    % 2. Calculate each parity bit as XOR of all positions it "covers"
    % 3. Parity bit at position p covers all positions where bit p is set in binary representation
    %
    % INPUTS:
    %   message - 11-bit information vector
    %   H - 4x15 parity check matrix
    %   info_positions - positions for information bits (0-indexed)
    %   parity_positions - positions for parity bits (0-indexed)
    %
    % OUTPUT:
    %   codeword - 15-bit encoded vector
    
    n = size(H, 2);  % codeword length (15)
    codeword = zeros(1, n);
    
    % Step 1: Place information bits in their designated positions
    codeword(info_positions) = message;
    
    % Step 2: Calculate parity bits using standard Hamming rule
    for i = 1:length(parity_positions)
        parity_pos = parity_positions(i);
        
        % Find all positions that this parity bit should cover
        % A position is covered if it has the parity bit set in its binary representation
        parity_bits = [];
        for pos = 1:n
            if bitand(pos, parity_pos) == parity_pos
                parity_bits = [parity_bits pos];
            end
        end
        
        % Remove the parity position itself from the list
        parity_bits = parity_bits(parity_bits ~= parity_pos);
        
        % Calculate parity as XOR (mod 2 sum) of all covered positions
        codeword(parity_pos) = mod(sum(codeword(parity_bits)), 2);
    end
end

%% Hamming decoder function (corrected)
function [decoded, error_detected, corrected_pos] = hamming_decode(received, H, info_positions)
    % Decode received codeword using syndrome decoding
    %
    % ALGORITHM:
    % 1. Calculate syndrome s = r * H^T (mod 2)
    % 2. Convert syndrome to decimal - this directly gives error position
    % 3. If syndrome = 0, no error detected
    % 4. If syndrome != 0, error at position = syndrome value (1-indexed)
    % 5. Correct error by flipping bit at detected position
    % 6. Extract information bits from corrected codeword
    %
    % INPUTS:
    %   received - 15-bit received vector (possibly corrupted)
    %   H - 4x15 parity check matrix
    %   info_positions - positions of information bits (0-indexed)
    %
    % OUTPUTS:
    %   decoded - 11-bit decoded message
    %   error_detected - 0 if no error, 1 if error detected and corrected
    %   corrected_pos - position where error was corrected (1-indexed), 0 if no error
    
    n = length(received);
    
    % Step 1: Calculate syndrome vector
    syndrome = mod(received * H', 2);
    
    % Step 2: Convert syndrome to decimal (error position indicator)
    % For Hamming codes, syndrome value directly gives error position
    syndrome_decimal = bi2de(syndrome, 'left-msb');
    
    corrected_pos = 0;
    error_detected = 0;
    
    if syndrome_decimal == 0
        % Step 3: No error detected - syndrome is zero
        decoded = received(info_positions);  % extract information bits directly
    else
        % Step 4-5: Error detected at position syndrome_decimal (1-indexed)
        error_detected = 1;
        corrected_pos = syndrome_decimal;
        
        % Correct the error by flipping the bit at detected position
        corrected = received;
        corrected(syndrome_decimal) = mod(corrected(syndrome_decimal) + 1, 2);
        
        % Step 6: Extract information bits from corrected codeword
        decoded = corrected(info_positions);
    end
end

%% test with simple example first
fprintf('\n=== Simple Test ===\n');
test_msg = [1 0 1 0 1 0 1 0 1 0 1]; % 11-bit message
test_encoded = hamming_encode(test_msg, H, info_positions, parity_positions);
fprintf('Message:  %s\n', mat2str(test_msg));
fprintf('Encoded:  %s\n', mat2str(test_encoded));

% introduce single error
corrupted = test_encoded;
error_pos = 7; % introduce error at position 7 (information bit position)
corrupted(error_pos) = mod(corrupted(error_pos) + 1, 2);
fprintf('Corrupted: %s (error at pos %d)\n', mat2str(corrupted), error_pos);

% decode
[decoded_msg, error_flag, corrected_pos] = hamming_decode(corrupted, H, info_positions);
fprintf('Decoded:  %s\n', mat2str(decoded_msg));
fprintf('Error detected: %d, Corrected position: %d\n', error_flag, corrected_pos);
fprintf('Decoding successful: %s\n', char('No' + 19*(isequal(test_msg, decoded_msg))));

% Test syndrome calculation manually
fprintf('\n=== Syndrome Verification ===\n');
syndrome_test = mod(corrupted * H', 2);
syndrome_val = bi2de(syndrome_test, 'left-msb');
fprintf('Calculated syndrome: %s (decimal: %d)\n', mat2str(syndrome_test), syndrome_val);
fprintf('Expected error position: %d, Detected: %d\n', error_pos, syndrome_val);

%% ber testing system
snr_db_range = 0:1:10;  % snr range in db
num_blocks = 1000;      % number of blocks to test

% storage for results
ber_uncoded = zeros(size(snr_db_range));
ber_hamming = zeros(size(snr_db_range));

fprintf('\n=== BER Testing ===\n');

for snr_idx = 1:length(snr_db_range)
    snr_db = snr_db_range(snr_idx);
    snr_linear = 10^(snr_db/10);
    noise_var = 1 / (2 * snr_linear);  % for bpsk
    
    % error counters
    errors_uncoded = 0;
    errors_hamming = 0;
    total_msg_bits = 0;
    
    for block = 1:num_blocks
        % generate random message
        message = randi([0 1], 1, k);
        
        % hamming encoding
        codeword = hamming_encode(message, H, info_positions, parity_positions);
        
        % bpsk modulation
        tx_uncoded = 2 * message - 1;      % uncoded bpsk
        tx_coded = 2 * codeword - 1;       % coded bpsk
        
        % awgn channel
        noise_uncoded = sqrt(noise_var) * randn(1, k);
        noise_coded = sqrt(noise_var) * randn(1, n);
        
        rx_uncoded = tx_uncoded + noise_uncoded;
        rx_coded = tx_coded + noise_coded;
        
        % bpsk demodulation (hard decision)
        demod_uncoded = rx_uncoded > 0;
        demod_coded = rx_coded > 0;
        
        % hamming decoding
        [decoded_message, ~, ~] = hamming_decode(demod_coded, H, info_positions);
        
        % count errors
        errors_uncoded = errors_uncoded + sum(message ~= demod_uncoded);
        errors_hamming = errors_hamming + sum(message ~= decoded_message);
        total_msg_bits = total_msg_bits + k;
    end
    
    % calculate ber
    ber_uncoded(snr_idx) = errors_uncoded / total_msg_bits;
    ber_hamming(snr_idx) = errors_hamming / total_msg_bits;
    
    fprintf('SNR = %2d dB: Uncoded BER = %.2e, Hamming BER = %.2e\n', ...
            snr_db, ber_uncoded(snr_idx), ber_hamming(snr_idx));
end

%% plot results
figure(1);
semilogy(snr_db_range, ber_uncoded, 'r-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
semilogy(snr_db_range, ber_hamming, 'b-s', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate');
title('BER Performance: Uncoded vs Hamming (15,11)');
legend('Uncoded BPSK', 'Hamming (15,11)', 'Location', 'southwest');
xlim([0 10]);
ylim([1e-5 1]);

%% calculate coding gain
fprintf('\n=== Performance Analysis ===\n');
idx_6db = find(snr_db_range == 6);
if ~isempty(idx_6db) && ber_uncoded(idx_6db) > 0 && ber_hamming(idx_6db) > 0
    coding_gain = 10*log10(ber_uncoded(idx_6db) / ber_hamming(idx_6db));
    fprintf('At SNR = 6 dB:\n');
    fprintf('  Uncoded BER: %.2e\n', ber_uncoded(idx_6db));
    fprintf('  Hamming BER: %.2e\n', ber_hamming(idx_6db));
    if ber_hamming(idx_6db) < ber_uncoded(idx_6db)
        fprintf('  Coding gain: %.1f dB\n', coding_gain);
    else
        fprintf('  Coding loss: %.1f dB\n', -coding_gain);
    end
end

% find crossover point
crossover_idx = find(ber_hamming < ber_uncoded, 1);
if ~isempty(crossover_idx)
    fprintf('Hamming code begins outperforming uncoded at SNR = %d dB\n', ...
            snr_db_range(crossover_idx));
end

fprintf('\nProblems and Benefits:\n');
fprintf('Benefits:\n');
fprintf('- Single error correction capability\n');
fprintf('- Systematic code (easy to implement)\n');
fprintf('- Good performance at moderate to high SNR\n');
fprintf('Problems:\n');
fprintf('- Code rate penalty (11/15 = 0.733)\n');
fprintf('- Cannot correct multiple errors\n');
fprintf('- May perform worse than uncoded at very low SNR due to rate loss\n');

%% error correction capability test
fprintf('\n=== Error Correction Test ===\n');
num_tests = 1000;
single_error_corrections = 0;

for test = 1:num_tests
    % random message and encoding
    msg = randi([0 1], 1, k);
    encoded = hamming_encode(msg, H, info_positions, parity_positions);
    
    % introduce single random error
    error_position = randi(n);
    corrupted = encoded;
    corrupted(error_position) = mod(corrupted(error_position) + 1, 2);
    
    % decode and check if corrected
    [decoded, ~, ~] = hamming_decode(corrupted, H, info_positions);
    if isequal(msg, decoded)
        single_error_corrections = single_error_corrections + 1;
    end
end

correction_rate = single_error_corrections / num_tests * 100;
fprintf('Single error correction rate: %.1f%% (%d/%d)\n', ...
        correction_rate, single_error_corrections, num_tests);

%% save results
save('exercise_4_1_1_results.mat', 'snr_db_range', 'ber_uncoded', 'ber_hamming', 'G', 'H', 'info_positions', 'parity_positions');
saveas(gcf, 'exercise_4_1_1_ber_comparison.png');

fprintf('\nExercise 4.1.1 completed successfully!\n');