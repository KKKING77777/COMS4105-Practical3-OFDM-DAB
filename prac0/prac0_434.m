function prac0_434
% Exercise 4.3.4 – BER over multipath (1080 MHz) with oversampling
% NRZ(BPSK) -> multipath (LOS + ground + mountain + mountain+ground)
% Bit rate 500 Mbit/s, oversample L samples/bit before the channel.

%% ------------ Parameters ------------
rng(1);
c   = 3e8;
fc  = 1080e6;                 % carrier for path phase
Rb  = 500e6;                  % bit rate (bits/s)
L   = 8;                      % samples per bit
Fs  = L*Rb;                   % sampling rate
Ts  = 1/Fs; Tb = 1/Rb;

Nbits   = 2e5;                % increase for smoother curves (slower)
EbN0dB  = 3:1:30;
EbN0    = 10.^(EbN0dB/10);

% Geometry (same as 4.3.2 landing)
xwall = 3000;
tx = [0,   951];
rx = [2700,450];

% Reflection coefficients
Gamma_g = -1.0;               % ground
Gamma_m = -0.8;               % mountain
Gamma_mg= Gamma_g*Gamma_m;

% Baseband NRZ rectangular pulse: make Eb = 1
A = sqrt(Rb);                 % energy Tb*A^2 = 1  => Eb = 1

%% ------------ Build multipath impulse response h[n] ------------
% Each path has complex gain including 1/R loss and exp(-j2πfcτ)
paths = {};

% LOS
[~,R_los] = one_len(tx,rx); tau_los = R_los/c;
a_los = (1/R_los);  c_los = a_los * exp(-1j*2*pi*fc*tau_los) * sgnc(1);
paths{end+1} = struct('tau',tau_los,'cplx',c_los);

% ground
tx_g = img_y(tx);
[~,R_g] = one_len(tx_g,rx); tau_g = R_g/c;
a_g  = (1/R_g)*abs(Gamma_g); c_g  = a_g*exp(-1j*2*pi*fc*tau_g)*sgnc(Gamma_g);
paths{end+1} = struct('tau',tau_g,'cplx',c_g);

% mountain
tx_m = img_x(tx,xwall);
[~,R_m] = one_len(tx_m,rx); tau_m = R_m/c;
a_m  = (1/R_m)*abs(Gamma_m); c_m  = a_m*exp(-1j*2*pi*fc*tau_m)*sgnc(Gamma_m);
paths{end+1} = struct('tau',tau_m,'cplx',c_m);

% mountain + ground
tx_mg = img_y(tx_m);
[~,R_mg] = one_len(tx_mg,rx); tau_mg = R_mg/c;
a_mg = (1/R_mg)*abs(Gamma_mg); c_mg = a_mg*exp(-1j*2*pi*fc*tau_mg)*sgnc(Gamma_mg);
paths{end+1} = struct('tau',tau_mg,'cplx',c_mg);

% Delay/ISI summary (增强 1)
taus = cellfun(@(p) p.tau, paths);
[~,~] = sort(taus,'ascend');
spread = max(taus) - min(taus);              % total delay spread
ISI_bits = spread / Tb;
fprintf('Earliest delay = %.3f us, latest = %.3f us, spread = %.3f us (~%.0f bits)\n', ...
        min(taus)*1e6, max(taus)*1e6, spread*1e6, ISI_bits);

% Build discrete-time composite h using fractional-delay FIR
M = 41;                               % odd length
Ntap = ceil(max(taus)*Fs) + M + 1;
h = zeros(1,Ntap);

for k = 1:numel(paths)
    d = paths{k}.tau * Fs;            % delay in samples
    n0 = floor(d);
    frac = d - n0;                    % 0..1
    hk = frac_delay_kernel(frac, M);  % length-M, unity DC gain
    start = n0 - floor(M/2) + 1;      % MATLAB indexing
    i1 = max(1, start);
    i2 = min(Ntap, start + M - 1);
    k1 = 1 + (i1 - start);
    k2 = M - ((start + M - 1) - i2);
    h(i1:i2) = h(i1:i2) + paths{k}.cplx * hk(k1:k2);
end

%% ------------ Generate data ------------
b = randi([0 1], Nbits, 1);
s_bit = 2*b - 1;                        % 0 -> -1, 1 -> +1
x = A * repelem(s_bit, L, 1);           % oversampled NRZ

%% ------------ BER loop (multipath + AWGN) ------------
BER = zeros(size(EbN0));
for m = 1:numel(EbN0)
    % channel
    y = conv(x, h, 'full');

    % complex AWGN: each real component var = N0/2*Fs  (Eb=1 => N0=1/EbN0)
    N0 = 1./EbN0(m);
    sigma = sqrt(N0*Fs/2);
    n = sigma*(randn(size(y)) + 1j*randn(size(y)));
    r = y + n;

    % integrate-and-dump (rectangular matched filter)
    start = ceil(min(taus)*Fs) + 1;
    stop  = start + Nbits*L - 1;
    if stop > numel(r), r = [r, zeros(1, stop - numel(r))]; end
    r_use = r(start:stop);
    r_mat = reshape(r_use, L, Nbits);
    z = sum(r_mat, 1) * Ts;
    bhat = real(z) > 0;
    BER(m) = nnz(bhat.' ~= b)/Nbits;

    fprintf('Multipath: Eb/N0 = %2d dB -> BER = %.3g\n', EbN0dB(m), BER(m));
end

%% ------------ AWGN-only baseline (增强 2) ------------
BER_awgn = zeros(size(EbN0));
for m = 1:numel(EbN0)
    N0 = 1./EbN0(m);
    sigma = sqrt(N0*Fs/2);
    n0 = sigma*(randn(size(x)) + 1j*randn(size(x)));
    r0 = x + n0;
    r0_mat = reshape(r0(1:Nbits*L), L, Nbits);
    z0 = sum(r0_mat,1)*Ts;
    bh0 = real(z0)>0;
    BER_awgn(m) = nnz(bh0.'~=b)/Nbits;
end

%% ------------ Plot ------------
figure('Name','Ex4.3.4 BER (multipath, 1080 MHz)');
semilogy(EbN0dB, BER, 'o-','LineWidth',1.5); hold on; grid on;
semilogy(EbN0dB, BER_awgn, '--','LineWidth',1.5);
ylim([1e-6 1]); xlim([min(EbN0dB) max(EbN0dB)]);
xlabel('E_b/N_0 (dB)'); ylabel('BER');
title(sprintf('BER with multipath @1080 MHz, R_b=%d Mb/s, L=%d', Rb/1e6, L));
legend('Multipath (ISI)','AWGN-only baseline','Location','southwest');

fprintf('\nNote: Path delays are microseconds vs Tb=2 ns -> heavy ISI; BER may floor near 0.5.\n');

end

%% ------------- helpers -------------
function h = frac_delay_kernel(frac, M)
% Windowed-sinc fractional delay FIR of length M (odd), delay = frac (0..1)
n = 0:M-1; m = (M-1)/2;
h = sinc((n-m) - frac);
w = hamming(M).';
h = h .* w;
h = h / sum(h);                 % unity DC gain
end

function [vec,R] = one_len(p1,p2), vec = p2-p1; R = hypot(vec(1),vec(2)); end
function p = img_y(p), p = [p(1), -p(2)]; end
function p = img_x(p,xw), p = [2*xw - p(1), p(2)]; end
function s = sgnc(G), if isreal(G), s = sign(G); else, s = G/abs(G); end, end
