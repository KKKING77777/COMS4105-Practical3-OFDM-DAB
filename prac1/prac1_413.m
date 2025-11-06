% prac1_413.m 
clear; clc; close all; rng(1);

Nbits = 2e4; sps = 8; fc = 0.2; A = 1; SNRdB = 10;

b   = randi([0 1], Nbits, 1);
sym = 2*b - 1;                   % BPSK ±1
x   = repelem(sym, sps);         % 矩形成形
n   = (0:numel(x)-1).';
c   = cos(2*pi*fc*n);
tx  = A*x .* c;

rx   = awgn(tx, SNRdB, 'measured');
i_bb = 2*rx .* c;                % 下变频I支路

h    = ones(sps,1);
y_mf = filter(h,1,i_bb)/sps;     % 匹配滤波，幅度≈±1
y_s  = y_mf(sps:sps:end);        % 每符号抽样（与眼图一致）
bhat = y_s >= 0;
BER  = mean(bhat ~= b);
fprintf('M1.3  SNR=%.1f dB, BER=%.3g\n', SNRdB, BER);

% 图1：通带PSD
figure;
[pxx,f] = pwelch(rx, [], [], [], 1, 'centered');
plot(f,10*log10(pxx)); grid on;
xlabel('Frequency (normalized)'); ylabel('PSD (dB)');
title('Received passband PSD');

% 图2：下变频I支路片段
figure;
seg = min(2000,numel(i_bb));
plot(i_bb(1:seg)); grid on;
xlabel('sample'); ylabel('I-channel voltage (real)');
title('I-branch waveform (downconverted segment)');

% 眼图：不要先 figure; 让 eyediagram 自建窗口（避免空白 Figure）
Leye = min(2*sps*800, numel(y_mf));
eyediagram(y_mf(1:Leye), 2*sps);
title('Eye diagram (2 symbols width, after matched filter)');

% 图4：统计
figure;
subplot(2,1,1);
histogram(y_s, 80); grid on;
xlabel('I-channel amplitude (post-integrate)'); ylabel('count');
title('I-channel amplitude histogram');
subplot(2,1,2);
plot(y_s); grid on;
xlabel('symbol index'); ylabel('Integrator output');
title('Integrator output (1 sample / symbol)');
