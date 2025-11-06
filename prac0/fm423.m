%% Practical 0 — Exercise 4.2.3  (FM signal, full & robust)
% %  250 kS/s    100.30 MHz
% ：pwelch -> fftshift(20*log10(...))

clear; close all; clc;

%% ========
fs = 250e3;                 
center_freq = 100.10e6;     
file = 'fm.bin';            

%% ========
fid = fopen(file, 'rb');
assert(fid >= 0, 'cannot open the file: %s', file);
raw = fread(fid, inf, 'float32=>double', 0, 'ieee-le');  % little-endian
fclose(fid);

if mod(numel(raw),2) ~= 0
    warning('If the number of samples is odd, discard the last sample to pair I/Q。');
    raw = raw(1:end-1);
end

raw(~isfinite(raw)) = 0;  % 清理 NaN/Inf
I = raw(1:2:end); Q = raw(2:2:end);
z = I + 1j*Q; clear raw I Q;

%% ========
Nwant = round(0.5*fs);
N = min(Nwant, numel(z));
z = z(1:N);
z = z - mean(z);                
z(~isfinite(real(z))) = 0;
z(~isfinite(imag(z))) = 0;

%% ========
nperseg = 4096;
if numel(z) < nperseg
    nperseg = max(256, 2.^floor(log2(max(256, numel(z)/4))));
end

%% ========
[fxx, f] = pwelch(z, [], [], [], fs);   % PSD
pxx_db = fftshift(20*log10(fxx + 1e-20));
freq_MHz = (f - mean(f))/1e6;           

%% ========
[peak_val, peak_idx] = max(pxx_db);
th = peak_val - 3;
mask = (pxx_db >= th);
L = peak_idx; R = peak_idx;
while L>1 && mask(L-1), L=L-1; end
while R<numel(mask) && mask(R+1), R=R+1; end

bw_hz = (freq_MHz(R) - freq_MHz(L)) * 1e6;     % -3 dB 
carrier_off_hz = freq_MHz(peak_idx) * 1e6;     
carrier_hz = center_freq + carrier_off_hz;     


guard = floor(0.5*(R-L+1));
valid = true(size(pxx_db));
valid(max(1,L-guard):min(numel(valid),R+guard)) = false;
noise_floor_db = median(pxx_db(valid));

df = mean(diff(f));                              % Hz/bin
sig_lin = 10.^(pxx_db(L:R)/10);                 
sig_power = sum(sig_lin) * df;
noise_lin = 10.^(noise_floor_db/10);
noise_power = noise_lin * ((R-L+1) * df);
snr_db = 10*log10(sig_power+eps) - 10*log10(noise_power+eps);

%% ==== figure ====
figure('Color','w','Name','FM PSD');
plot(freq_MHz, pxx_db, 'LineWidth', 1.1); grid on; hold on;
xlabel('Frequency offset from centre (MHz)');
ylabel('Power (dB)');
title(sprintf('FM Welch PSD (0.5 s, fs=%.3f MS/s)', fs/1e6));

xline(freq_MHz(peak_idx),'--','Carrier','LabelOrientation','horizontal');
xline(freq_MHz(L),':','-3 dB edge');
xline(freq_MHz(R),':','-3 dB edge');

info = sprintf(['Carrier: %.6f MHz (offset %.1f kHz)\n' ...
                'BW(-3 dB): %.1f kHz\nNoise floor: %.1f dB\nSNR: %.1f dB'], ...
                carrier_hz/1e6, carrier_off_hz/1e3, bw_hz/1e3, noise_floor_db, snr_db);
xl = xlim; yl = ylim;
x_range = max(xl) - min(xl);
y_range = max(yl) - min(yl);
text(xl(1)+0.02*x_range, yl(1)+0.06*y_range, info, ...
    'BackgroundColor',[1 1 1 0.85], 'Margin',6, 'EdgeColor',[0.7 0.7 0.7]);

%% ====print====
fprintf('\n[FM capture results]\n');
fprintf('Carrier: %.6f MHz (offset %.1f kHz)\n', carrier_hz/1e6, carrier_off_hz/1e3);
fprintf('Bandwidth (-3 dB): %.1f kHz\n', bw_hz/1e3);
fprintf('Noise floor: %.1f dB | SNR: %.1f dB\n\n', noise_floor_db, snr_db);
