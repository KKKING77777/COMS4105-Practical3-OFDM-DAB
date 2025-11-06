rng(7);
numBits = 20;
sps = 40;                 % samples per bit
noiseVar = 0.5;           %  2^-1
sigma = sqrt(noiseVar);

% Transimitter: Polar NRZ
bits = randi([0 1], numBits, 1);
symbols = 2*bits - 1;
tx = repelem(symbols, sps);

% Chanel：AWGN
rx = tx + sigma*randn(size(tx));

% Receiver
rxMat = reshape(rx, sps, numBits).';
avg = mean(rxMat, 2);
dec = avg > 0;
BER = mean(dec ~= bits);

% Plot waveform
figure;
plot(rx); hold on;
for k = 1:numBits-1, xline(k*sps,'--'); end
title('Received waveform: Polar NRZ over AWGN (noise power = 1/2)');
xlabel('Sample index'); ylabel('Amplitude'); grid on;

% Print to console
disp('Bits:'); disp(bits.');
disp('Decisions:'); disp(dec.');
fprintf(' BER = %.4f\n', BER);
