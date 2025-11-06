%% COMS4105 Practical 3 - Exercise 4.1.2
% frequency estimation and recovery
% frequency shift simulation

clear; close all; clc;

% load from 4.1.1
load('exercise_4_1_1_results.mat');

fprintf('=== Exercise 4.1.2: Frequency Estimation ===\n');

%% generate test signal
test_signal = ofdm_signal(:, 1);  % first symbol
useful_part = test_signal(N_guard+1:end);  % no prefix

%% apply freq shift
max_freq_shift = 5;  % max shift carriers
actual_freq_shift = round((rand - 0.5) * 2 * max_freq_shift);  % random
fprintf('Applied frequency shift: %d subcarriers (%.2f kHz)\n', ...
        actual_freq_shift, actual_freq_shift * subcarrier_spacing / 1000);

% apply shift via circshift
freq_spectrum_original = fft(useful_part, N_fft);
freq_spectrum_shifted = circshift(freq_spectrum_original, actual_freq_shift);
shifted_time_signal = ifft(freq_spectrum_shifted, N_fft);

% add prefix back
shifted_ofdm = [shifted_time_signal(end-N_guard+1:end); shifted_time_signal];

%% freq estimation algorithm
function estimated_shift = coarse_freq_estimate(signal, N_fft, N_guard, N_active)
    % remove prefix
    useful_signal = signal(N_guard+1:end);
    
    % fft
    freq_domain = fft(useful_signal, N_fft);
    power_spectrum = abs(freq_domain).^2;
    
    % search range
    max_search = 10;  % ±10 carriers
    search_range = -max_search:max_search;
    
    % power scores
    power_scores = zeros(size(search_range));
    
    % expected active carriers
    expected_active = (-N_active/2:N_active/2-1) + N_fft/2 + 1;
    
    % bounds check
    expected_active = expected_active(expected_active >= 1 & expected_active <= N_fft);
    
    for i = 1:length(search_range)
        shift = search_range(i);
        
        % trial shift
        shifted_power = circshift(power_spectrum, shift);
        
        % active power
        active_power = sum(shifted_power(expected_active));
        
        % total power
        total_power = sum(shifted_power);
        
        % power ratio score
        power_scores(i) = active_power / (total_power + eps);
    end
    
    % max score
    [~, max_idx] = max(power_scores);
    estimated_shift = search_range(max_idx);
end

%% estimate and correct
estimated_shift = coarse_freq_estimate(shifted_ofdm, N_fft, N_guard, N_active);
fprintf('Estimated frequency shift: %d subcarriers\n', estimated_shift);
fprintf('Estimation error: %d subcarriers\n', abs(actual_freq_shift - estimated_shift));

% apply correction
corrected_useful = shifted_ofdm(N_guard+1:end);
freq_corrected = fft(corrected_useful, N_fft);
freq_corrected = circshift(freq_corrected, -estimated_shift);
corrected_time = ifft(freq_corrected, N_fft);
corrected_ofdm = [corrected_time(end-N_guard+1:end); corrected_time];

%% performance comparison
% extract active carriers
active_indices = (-N_active/2:N_active/2-1) + N_fft/2 + 1;

% original
original_freq = fft(useful_part, N_fft);
original_active = original_freq(active_indices);

% shifted
shifted_freq = fft(shifted_ofdm(N_guard+1:end), N_fft);
shifted_active = shifted_freq(active_indices);

% corrected
corrected_freq = fft(corrected_ofdm(N_guard+1:end), N_fft);
corrected_active = corrected_freq(active_indices);

%% plot results
figure(1);
subplot(2,3,1);
freq_axis = (-N_fft/2:N_fft/2-1) * subcarrier_spacing / 1000;
plot(freq_axis, fftshift(abs(original_freq)));
title('Original Signal Spectrum');
xlabel('Frequency (kHz)');
ylabel('Magnitude');
grid on;

subplot(2,3,2);
plot(freq_axis, fftshift(abs(shifted_freq)));
title(sprintf('Shifted Signal (shift=%d)', actual_freq_shift));
xlabel('Frequency (kHz)');
ylabel('Magnitude');
grid on;

subplot(2,3,3);
plot(freq_axis, fftshift(abs(corrected_freq)));
title(sprintf('Corrected Signal (est=%d)', estimated_shift));
xlabel('Frequency (kHz)');
ylabel('Magnitude');
grid on;

% constellation plots
subplot(2,3,4);
scatter(real(original_active), imag(original_active), 'bo');
title('Original Constellation');
xlabel('In-phase');
ylabel('Quadrature');
grid on;
axis equal;

subplot(2,3,5);
scatter(real(shifted_active), imag(shifted_active), 'ro');
title('Shifted Constellation');
xlabel('In-phase');
ylabel('Quadrature');
grid on;
axis equal;

subplot(2,3,6);
scatter(real(corrected_active), imag(corrected_active), 'go');
title('Corrected Constellation');
xlabel('In-phase');
ylabel('Quadrature');
grid on;
axis equal;

%% correction performance
% normalize for comparison
orig_power = sqrt(mean(abs(original_active).^2));
shifted_power = sqrt(mean(abs(shifted_active).^2));
corrected_power = sqrt(mean(abs(corrected_active).^2));

% calculate evm
shifted_normalized = shifted_active * orig_power / shifted_power;
corrected_normalized = corrected_active * orig_power / corrected_power;

evm_shifted = sqrt(mean(abs(original_active - shifted_normalized).^2)) / orig_power * 100;
evm_corrected = sqrt(mean(abs(original_active - corrected_normalized).^2)) / orig_power * 100;

fprintf('\n=== Performance Results ===\n');
fprintf('EVM before correction: %.2f%%\n', evm_shifted);
fprintf('EVM after correction: %.2f%%\n', evm_corrected);
fprintf('Improvement: %.2f%%\n', evm_shifted - evm_corrected);

%% test multiple shifts
fprintf('\n=== Testing Multiple Random Shifts ===\n');
num_tests = 50;
test_shifts = randi([-8, 8], 1, num_tests);
estimation_errors = zeros(1, num_tests);

for test_idx = 1:num_tests
    % apply test shift
    test_freq = circshift(freq_spectrum_original, test_shifts(test_idx));
    test_time = ifft(test_freq, N_fft);
    test_ofdm = [test_time(end-N_guard+1:end); test_time];
    
    % estimate shift
    est_shift = coarse_freq_estimate(test_ofdm, N_fft, N_guard, N_active);
    estimation_errors(test_idx) = abs(test_shifts(test_idx) - est_shift);
end

perfect_estimates = sum(estimation_errors == 0);
success_rate = perfect_estimates / num_tests * 100;

fprintf('Perfect estimation rate: %.1f%% (%d out of %d)\n', ...
        success_rate, perfect_estimates, num_tests);
fprintf('Mean estimation error: %.2f subcarriers\n', mean(estimation_errors));
fprintf('Max estimation error: %d subcarriers\n', max(estimation_errors));

%% Alternative Frequency Estimation Methods
% Method 2: Cross-correlation with ideal spectrum
function estimated_shift_xcorr = xcorr_freq_estimate(signal, reference_spectrum, N_fft, N_guard)
    % Remove cyclic prefix and take FFT
    useful_signal = signal(N_guard+1:end);
    freq_domain = abs(fft(useful_signal, N_fft));
    
    % Cross-correlate power spectra
    ref_power = abs(reference_spectrum);
    
    % Circular cross-correlation
    xcorr_result = zeros(1, N_fft);
    for shift = 0:N_fft-1
        shifted_ref = circshift(ref_power, shift);
        xcorr_result(shift+1) = sum(freq_domain .* shifted_ref);
    end
    
    % Find peak (convert to signed shift)
    [~, max_idx] = max(xcorr_result);
    estimated_shift_xcorr = max_idx - 1;
    
    % Convert to symmetric range [-N_fft/2, N_fft/2-1]
    if estimated_shift_xcorr > N_fft/2
        estimated_shift_xcorr = estimated_shift_xcorr - N_fft;
    end
end

% Test alternative method
alt_estimate = xcorr_freq_estimate(shifted_ofdm, freq_spectrum_original, N_fft, N_guard);
fprintf('\nAlternative method estimate: %d subcarriers\n', alt_estimate);
fprintf('Alternative method error: %d subcarriers\n', abs(actual_freq_shift - alt_estimate));

%% Save results for next exercise
save('exercise_4_1_2_results.mat', 'shifted_ofdm', 'corrected_ofdm', 'actual_freq_shift', ...
     'estimated_shift', 'original_freq', 'shifted_freq', 'corrected_freq');

% Save figure
saveas(gcf, 'exercise_4_1_2_results.png');
fprintf('\nFigure saved as exercise_4_1_2_results.png\n');

fprintf('\nExercise 4.1.2 completed successfully!\n');