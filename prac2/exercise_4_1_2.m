%% COMS4105 Practical 2 - Exercise 4.1.2
% crc-16 error detection system
% dab crc-16 implementation

clear; close all; clc;

%% crc-16 parameters (dab standard)
% polynomial: x^16 + x^12 + x^5 + 1
% binary: 1 0001 0000 0010 0001
crc_poly = [1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1]; % generator polynomial
message_bits = 16;  % message block size
crc_bits = 16;      % crc checksum size
total_bits = message_bits + crc_bits;

fprintf('=== CRC-16 Error Detection System ===\n');
fprintf('Generator polynomial: x^16 + x^12 + x^5 + 1\n');
fprintf('Message bits per block: %d\n', message_bits);
fprintf('CRC bits: %d\n', crc_bits);
fprintf('Total frame bits: %d\n', total_bits);

%% crc-16 encoder function
function crc = crc16_encode(message)
    % dab crc-16 encoder
    poly = [1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1];
    
    % append 16 zeros for division
    dividend = [message zeros(1, 16)];
    
    % polynomial long division
    for i = 1:(length(message))
        if dividend(i) == 1
            for j = 1:length(poly)
                dividend(i + j - 1) = xor(dividend(i + j - 1), poly(j));
            end
        end
    end
    
    % crc is remainder
    crc = dividend(end-15:end);
end

%% crc-16 checker function
function error_detected = crc16_check(received_frame)
    % check if crc is valid
    poly = [1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1];
    
    % polynomial division on entire frame
    dividend = received_frame;
    
    for i = 1:(length(received_frame) - 16)
        if dividend(i) == 1
            for j = 1:length(poly)
                if (i + j - 1) <= length(dividend)
                    dividend(i + j - 1) = xor(dividend(i + j - 1), poly(j));
                end
            end
        end
    end
    
    % if remainder is zero, no error detected
    remainder = dividend(end-15:end);
    error_detected = any(remainder);
end

%% error detection performance test
snr_db_range = 0:1:12;  % snr range
num_frames = 10000;     % frames per snr point

% storage for results
frames_with_errors = zeros(size(snr_db_range));
detected_errors = zeros(size(snr_db_range));
undetected_errors = zeros(size(snr_db_range));

fprintf('\n=== Error Detection Performance Test ===\n');

for snr_idx = 1:length(snr_db_range)
    snr_db = snr_db_range(snr_idx);
    snr_linear = 10^(snr_db/10);
    noise_var = 1 / (2 * snr_linear);  % for bpsk
    
    error_frames = 0;
    detected_count = 0;
    undetected_count = 0;
    
    for frame = 1:num_frames
        % generate random message
        message = randi([0 1], 1, message_bits);
        
        % crc encoding
        crc = crc16_encode(message);
        tx_frame = [message crc];
        
        % bpsk modulation and awgn channel
        tx_signal = 2 * tx_frame - 1;
        noise = sqrt(noise_var) * randn(1, total_bits);
        rx_signal = tx_signal + noise;
        
        % bpsk demodulation
        rx_frame = rx_signal > 0;
        
        % check for transmission errors
        frame_has_errors = any(tx_frame ~= rx_frame);
        
        if frame_has_errors
            error_frames = error_frames + 1;
            
            % check if crc detects the error
            if crc16_check(rx_frame)
                detected_count = detected_count + 1;
            else
                undetected_count = undetected_count + 1;
            end
        end
    end
    
    frames_with_errors(snr_idx) = error_frames;
    detected_errors(snr_idx) = detected_count;
    undetected_errors(snr_idx) = undetected_count;
    
    detection_rate = detected_count / max(error_frames, 1) * 100;
    
    fprintf('SNR = %2d dB: Frames with errors = %4d, Detected = %4d, Undetected = %4d (%.1f%% detection)\n', ...
            snr_db, error_frames, detected_count, undetected_count, detection_rate);
end

%% plot results
figure(1);
subplot(2,1,1);
semilogy(snr_db_range, frames_with_errors, 'r-o', 'LineWidth', 2);
hold on;
semilogy(snr_db_range, detected_errors, 'g-s', 'LineWidth', 2);
semilogy(snr_db_range, undetected_errors, 'b-^', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('Number of Frames');
title('CRC-16 Error Detection Performance');
legend('Total Error Frames', 'Detected Errors', 'Undetected Errors', 'Location', 'northeast');

subplot(2,1,2);
detection_rate = detected_errors ./ max(frames_with_errors, 1) * 100;
plot(snr_db_range, detection_rate, 'k-o', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('SNR (dB)');
ylabel('Detection Rate (%)');
title('CRC-16 Error Detection Rate vs SNR');
ylim([0 105]);

%% test with specific example from preparation
fprintf('\n=== Preparation Question Test ===\n');
test_message = [1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1]; % from prep question
test_crc = crc16_encode(test_message);
fprintf('Message: %s\n', mat2str(test_message));
fprintf('CRC-16:  %s\n', mat2str(test_crc));

% verify encoding/decoding
test_frame = [test_message test_crc];
error_check = crc16_check(test_frame);
fprintf('CRC check result: %d (0 = no error detected)\n', error_check);

%% analysis
fprintf('\n=== Performance Analysis ===\n');
fprintf('CRC-16 Properties:\n');
fprintf('- Detects all single-bit errors\n');
fprintf('- Detects all double-bit errors\n');
fprintf('- Detects all burst errors <= 16 bits\n');
fprintf('- Detects 99.998%% of longer burst errors\n');

high_snr_idx = find(snr_db_range >= 8);
if ~isempty(high_snr_idx)
    avg_detection_rate = mean(detection_rate(high_snr_idx));
    fprintf('Average detection rate at high SNR (>= 8 dB): %.1f%%\n', avg_detection_rate);
end

%% save results
save('exercise_4_1_2_results.mat', 'snr_db_range', 'frames_with_errors', ...
     'detected_errors', 'undetected_errors', 'detection_rate');
saveas(gcf, 'exercise_4_1_2_crc_performance.png');

fprintf('\nExercise 4.1.2 completed successfully!\n');