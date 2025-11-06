% ==== DAB  PSD  ====
clear; clc;

% ==== SETTINGS ====
filename = 'dab.bin';   % DAB.bin
fs = 2e6;               % DAB SR 2 MS/s
fc = 220e6;             % SET 220 MHz 

% ====  IQ DATA ====
fid = fopen(filename,'rb');
if fid < 0
    error('error', filename);
end
raw = fread(fid, inf, 'uint8=>single'); fclose(fid);
raw = (raw - 127.5) / 128.0;   %  -1~+1
I = raw(1:2:end); 
Q = raw(2:2:end);
x = complex(I, Q);

% ==== FFT ====
N = length(x);
X = fftshift(fft(x, N));          % center
Pxx = abs(X).^2 / N;              % 
Pxx_dB = 10*log10(Pxx + eps);     % turn to dB

% ==== frequency ====
f = (-N/2:N/2-1)*(fs/N);          %  Hz

% ==== plot ====
plot(f/1e3, Pxx_dB);  %  kHz
xlabel('Frequency offset (kHz)');
ylabel('Power (dB)');
title(['DAB Spectrum around ', num2str(fc/1e6), ' MHz']);
grid on;
