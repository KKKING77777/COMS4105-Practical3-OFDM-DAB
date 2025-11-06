function prac0_432
% Exercise 4.3.2 - link at 1080 MHz, landing point change and explanation

% ---- constants
c  = 3e8;
x_wall = 3000;

% ---- reflection coefficients (same as 4.3.1 unless specified)
Gamma_g  = -1.0;
Gamma_m  = -0.8;
Gamma_mg = Gamma_g * Gamma_m;

% ---- TX & RX (landing case)
tx = [0, 951];
rx = [2700, 450];

% ---- frequency range around 1080 MHz (optional wide view)
f = (1000:1:1100)*1e6;

% ---- compute response and plot
H = channel_freq_response(tx, rx, f, x_wall, Gamma_g, Gamma_m, Gamma_mg, c);
figure('Name','Ex4.3.2 Landing'); 
plot(f/1e6, 20*log10(abs(H)),'LineWidth',1.5); grid on;
xlabel('Frequency (MHz)'); ylabel('|H(f)| (dB)');
title('Exercise 4.3.2: Channel at landing point (2700,450)');

% ---- path-lengths and phase at 1080 MHz
[~,L_los] = one_len(tx, rx);
[~,L_mtn] = one_len(img_x(tx,x_wall), rx);
dL = L_mtn - L_los;
lambda = c/1080e6;
phi_deg = rad2deg(mod(2*pi*dL/lambda, 2*pi));

fprintf('[Ex4.3.2] LOS = %.1f m, Mountain = %.1f m, ΔL = %.1f m\n', L_los, L_mtn, dL);
fprintf('[Ex4.3.2] At 1080 MHz: phase diff ≈ %.1f deg\n', phi_deg);
fprintf('If phase ≈ (2k+1)*180° and amplitudes similar -> deep fade.\n');

% ---- notch spacing estimate and simple mitigations
delta_f = c/abs(dL);      % Hz
fprintf('Notch spacing estimate: Δf ≈ c/ΔL = %.1f MHz\n', delta_f/1e6);
fprintf('Mitigation 1: frequency diversity (shift a few MHz from 1080).\n');
fprintf('Mitigation 2: position/height diversity (move RX or change height).\n');

% ---- quick sensitivity to small height change (optional overlay)
rx_up = rx + [0, +1];
rx_dn = rx + [0, -1];
H_up = channel_freq_response(tx, rx_up, f, x_wall, Gamma_g, Gamma_m, Gamma_mg, c);
H_dn = channel_freq_response(tx, rx_dn, f, x_wall, Gamma_g, Gamma_m, Gamma_mg, c);
% hold on; plot(f/1e6, 20*log10(abs(H_up)),'--'); plot(f/1e6, 20*log10(abs(H_dn)),':');
% legend('base','y+1 m','y-1 m');

end

% ================= helper functions (local) =================
function H = channel_freq_response(tx, rx, f, xwall, Gg, Gm, Gmg, c)
L = numel(f); H = zeros(1,L);
tx_g  = img_y(tx);
tx_m  = img_x(tx, xwall);
tx_mg = img_y(tx_m);

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
function p = img_y(p), p = [p(1), -p(2)]; end
function p = img_x(p,xw), p = [2*xw - p(1), p(2)]; end
