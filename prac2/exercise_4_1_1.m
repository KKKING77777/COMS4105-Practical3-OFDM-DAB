%% COMS4105 Practical 2 - Exercise 4.1.1
% hamming (15,11) encoder/decoder with ber testing
% channel coding performance evaluation

clear; close all; clc;

%% hamming (15,11) parameters
n = 15; % codeword length
k = 11; % message length
t = 1;  % error correction capability

% generator matrix (systematic form)
% standard (15,11) hamming code generator
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

% parity check matrix
H = [0 1 1 1 1 0 1 0 1 0 0 1 0 0 0;
     1 0 1 1 1 1 0 1 0 1 0 0 1 0 0;
     1 1 0 1 1 1 1 0 1 0 1 0 0 1 0;
     1 1 1 0 1 1 1 1 0 1 0 0 0 0 1];

fprintf('=== Hamming (15,11) Code Parameters ===\n');
fprintf('Code length n = %d\n', n);
fprintf('Message length k = %d\n', k);
fprintf('Code rate = %.3f\n', k/n);
fprintf('Error correction capability t = %d\n', t);

%% hamming encoder function
function codeword = hamming_encode(message, G)
    % encode k-bit message to n-bit codeword
    codeword = mod(message * G, 2);
end

%% hamming decoder function  
function [decoded, error_detected] = hamming_decode(received, H, k)
    n = length(received);
    
    % calculate syndrome
    syndrome = mod(received * H', 2);
    
    % check for errors
    if sum(syndrome) == 0
        % no error detected
        decoded = received(1:k);  % extract message bits
        error_detected = 0;
    else
        % error detected, find error position
        syndrome_decimal = bi2de(syndrome, 'left-msb');
        
        if syndrome_decimal <= n
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

%% ber testing system
snr_db_range = 0:1:10;  % snr range in db
num_bits = 100000;      % total bits to test
num_blocks = ceil(num_bits / k);

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
    total_bits_sent = 0;
    
    for block = 1:num_blocks
        % generate random message
        message = randi([0 1], 1, k);
        
        % hamming encoding
        codeword = hamming_encode(message, G);
        
        % bpsk modulation
        tx_uncoded = 2 * message - 1;      % uncoded bpsk
        tx_coded = 2 * codeword - 1;       % coded bpsk
        
        % awgn channel
        noise_uncoded = sqrt(noise_var) * randn(1, k);
        noise_coded = sqrt(noise_var) * randn(1, n);
        
        rx_uncoded = tx_uncoded + noise_uncoded;
        rx_coded = tx_coded + noise_coded;
        
        % bpsk demodulation
        demod_uncoded = rx_uncoded > 0;
        demod_coded = rx_coded > 0;
        
        % hamming decoding
        [decoded_message, ~] = hamming_decode(demod_coded, H, k);
        
        % count errors
        errors_uncoded = errors_uncoded + sum(message ~= demod_uncoded);
        errors_hamming = errors_hamming + sum(message ~= decoded_message);
        total_bits_sent = total_bits_sent + k;
    end
    
    % calculate ber
    ber_uncoded(snr_idx) = errors_uncoded / total_bits_sent;
    ber_hamming(snr_idx) = errors_hamming / total_bits_sent;
    
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
ylim([1e-6 1]);

%% analysis
fprintf('\n=== Performance Analysis ===\n');
fprintf('At SNR = 6 dB:\n');
idx_6db = find(snr_db_range == 6);
if ~isempty(idx_6db)
    coding_gain = 10*log10(ber_uncoded(idx_6db) / ber_hamming(idx_6db));
    fprintf('  Uncoded BER: %.2e\n', ber_uncoded(idx_6db));
    fprintf('  Hamming BER: %.2e\n', ber_hamming(idx_6db));
    fprintf('  Coding gain: %.1f dB\n', coding_gain);
end

fprintf('\nProblems and Benefits:\n');
fprintf('Benefits:\n');
fprintf('- Single error correction capability\n');
fprintf('- Systematic code (easy to implement)\n');
fprintf('- Significant coding gain at moderate SNR\n');
fprintf('Problems:\n');
fprintf('- Code rate penalty (11/15 = 0.733)\n');
fprintf('- Cannot correct multiple errors\n');
fprintf('- Performance degrades at very low SNR\n');

%% save results
save('exercise_4_1_1_results.mat', 'snr_db_range', 'ber_uncoded', 'ber_hamming', 'G', 'H');
saveas(gcf, 'exercise_4_1_1_ber_comparison.png');

fprintf('\nExercise 4.1.1 completed successfully!\n');