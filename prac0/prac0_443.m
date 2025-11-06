function prac0_443
% Ex4.4.3 — Analytical PSD of Nxx1 vs Welch estimate

Rb = 1e6;                 % bit rate
L  = 32;                  % samples/bit (>=16)
Fs = L*Rb;
Nbits = 1e5;

% --- simulate waveform
b = randi([0 1], Nbits, 1);
x = nxx1_encode(b, L);                % +/-1
nfft = 16384; win = 8192; olap = win/2;
[PX,F] = pwelch(x, win, olap, nfft, Fs, 'centered');   % PSD (per Hz)

% --- analytical PSD (shape)
Tb = 1/Rb;
G = Tb .* sinc(0.5*F*Tb).^2 .* sin((pi/2)*F*Tb).^2;    % A=1

% --- normalize for shape comparison
PXdB = 10*log10(PX./max(PX));
GdB  = 10*log10(G ./max(G));

Fn = F/Rb;
figure('Name','Ex4.4.3 PSD (Nxx1)');
plot(Fn, PXdB, 'LineWidth',1.3); hold on; grid on;
plot(Fn, GdB,  '--', 'LineWidth',1.6);
xlabel('f / R_b'); ylabel('PSD (dB, normalized)');
title(sprintf('Nxx1 PSD: Analytical vs Welch (Nbits=%d, L=%d)', Nbits, L));
legend('Welch (simulated)','Analytical', 'Location','southwest');
xlim([-4 4]);
end

function y = nxx1_encode(b,L)
lev = -1; N = numel(b); y = zeros(N*L,1);
for k = 1:N
    lev = -lev;
    y((k-1)*L + (1:L/2)) = lev;
    if b(k)==0, lev = -lev; end
    y((k-1)*L + (L/2+1:L)) = lev;
end
end
