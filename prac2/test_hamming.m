%% Quick test of Hamming code - minimal version
clear; clc;

% Simple (7,4) Hamming for verification
fprintf('Testing basic Hamming concepts...\n');

% Test 1: Simple encoding/decoding
msg = [1 0 1 1];
fprintf('Message: %s\n', mat2str(msg));

% Basic Hamming (7,4) - well known
G = [1 0 0 0 1 1 0;
     0 1 0 0 1 0 1;
     0 0 1 0 0 1 1;
     0 0 0 1 1 1 1];

encoded = mod(msg * G, 2);
fprintf('Encoded: %s\n', mat2str(encoded));

H = [1 1 0 1 1 0 0;
     1 0 1 1 0 1 0;
     0 1 1 1 0 0 1];

% Test syndrome
syndrome = mod(encoded * H', 2);
fprintf('Syndrome (should be zero): %s\n', mat2str(syndrome));

% Introduce error
corrupted = encoded;
corrupted(3) = ~corrupted(3);
fprintf('Corrupted: %s\n', mat2str(corrupted));

% Calculate syndrome
error_syndrome = mod(corrupted * H', 2);
error_pos = bi2de(error_syndrome, 'left-msb');
fprintf('Error syndrome: %s (position %d)\n', mat2str(error_syndrome), error_pos);

% Correct error
corrected = corrupted;
if error_pos > 0 && error_pos <= 7
    corrected(error_pos) = ~corrected(error_pos);
end
fprintf('Corrected: %s\n', mat2str(corrected));
fprintf('Success: %s\n', char('No' + (isequal(encoded, corrected)) * 3));

fprintf('\nBasic Hamming test completed!\n');