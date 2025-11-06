function prac0_431
% Exercise 4.3.1 - channel magnitude vs frequency (1000–1100 MHz)

% ---- constants
c  = 3e8;          % speed of light (m/s)
x_wall = 3000;     % mountain wall at x = 3000 m

% ---- reflection coefficients (tune if your sheet gives values)
Gamma_g  = -1.0;   % ground reflection
Gamma_m  = -0.8;   % mountain reflection
Gamma_mg = Gamma_g * Gamma_m;

% ---- TX & RX positions (m). Change if your sheet uses others.
tx = [0, 951];
rx = [3200, 951];

% ---- frequency vector 1000:1:1100 MHz
f = (1000:1:1100)*1e6;

% ---- complex frequency response
H = channel_freq_response(tx, rx, f, x_wall, Gamma_g, Gamma_m, Gamma_mg, c);

% ---- plot |H|
figure('Name','Ex4.3.1 Magnitude'); 
plot(f/1e6, 20*log10(abs(H)),'LineWidth',1.5); grid on;
xlabel('Frequency (MHz)'); ylabel('|H(f)| (dB)');
title('Exercise 4.3.1: Magnitude response (1000–1100 MHz)');

% ---- readout around 1080 MHz
[~,i1080] = min(abs(f-1080e6));
p1080 = 20*log10(abs(H(i1080)));
win = 3; idx = max(1,i1080-win):min(numel(f),i1080+win);
padj = mean(20*log10(abs(H(idx))));
fprintf('[Ex4.3.1] |H| at 1080 MHz = %.2f dB; local avg(±3 MHz) = %.2f dB\n', p1080, padj);
if p1080 < padj - 2
    fprintf('Comment: 1080 MHz is in a fade notch (destructive multipath).\n');
else
    fprintf('Comment: 1080 MHz is similar to adjacent bins.\n');
end
end

% ================= helper functions (local) =================
function H = channel_freq_response(tx, rx, f, xwall, Gg, Gm, Gmg, c)
L = numel(f); H = zeros(1,L);
tx_g  = img_y(tx);              % ground image
tx_m  = img_x(tx, xwall);       % mountain image
tx_mg = img_y(tx_m);            % mountain + ground

[~,R_los] = one_len(tx,    rx); a_los = 1./R_los;
[~,R_g  ] = one_len(tx_g,  rx); a_g   = abs(Gg) ./R_g  .* phsgn(Gg);
[~,R_m  ] = one_len(tx_m,  rx); a_m   = abs(Gm) ./R_m  .* phsgn(Gm);
[~,R_mg ] = one_len(tx_mg, rx); a_mg  = abs(Gmg)./R_mg .* phsgn(Gmg);

for k = 1:L
    w = 2*pi*f(k);
    H(k) = a_los*exp(-1j*w*R_los/c) + ...
           a_g  *exp(-1j*w*R_g  /c) + ...
           a_m  *exp(-1j*w*R_m  /c) + ...
           a_mg *exp(-1j*w*R_mg /c);
end
end

function s = phsgn(G)
if isreal(G), s = sign(G); else, s = G/abs(G); end
end
function [vec,R] = one_len(p1,p2), vec = p2-p1; R = hypot(vec(1),vec(2)); end
function p = img_y(p), p = [p(1), -p(2)]; end          % reflect over y=0
function p = img_x(p,xw), p = [2*xw - p(1), p(2)]; end % reflect over x=xw
