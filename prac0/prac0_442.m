function prac0_442
% Ex4.4.2 — Welch PSD of Nxx1 line code

Rb = 1e6;          % bit rate (scale)
L  = 16;           % samples/bit (>=8 shows clear nulls)
Fs = L*Rb;

Nbits = 1e5;                          % >1000
b = randi([0 1], Nbits, 1);
x = nxx1_encode(b, L);                % +/-1 waveform

nfft = 8192; win = 4096; olap = win/2;
[PX,F] = pwelch(x, win, olap, nfft, Fs, 'centered');

Fnorm = F / Rb;                       % normalize freq by Rb
figure('Name','Ex4.4.2 PSD');
plot(Fnorm, 10*log10(PX), 'LineWidth',1.2); grid on;
xlabel('f / R_b'); ylabel('PSD (dB/Hz)');
title(sprintf('Welch PSD of Nxx1 (Nbits=%d, L=%d, nfft=%d)', Nbits, L, nfft));
xlim([-4 4]);
end

function y = nxx1_encode(b,L)
% Transition at bit start; mid-bit transition for 0; none for 1
lev = -1; N = numel(b); y = zeros(N*L,1);
for k = 1:N
    lev = -lev;
    y((k-1)*L + (1:L/2)) = lev;
    if b(k)==0, lev = -lev; end
    y((k-1)*L + (L/2+1:L)) = lev;
end
end
