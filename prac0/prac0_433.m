function prac0_433
% Exercise 4.3.3 – BER of NRZ (BPSK) over AWGN vs Eb/N0
% Simple one-sample-per-bit model (as wiki note: randn noise power=1 per symbol sample)

rng(1);                           % reproducible
EbN0dB = 3:1:30;                  % x-axis (dB)
EbN0   = 10.^(EbN0dB/10);         % linear

Nbits  = 2e6;                     % > 1e6 bits as requested
A      = 1;                       % bipolar NRZ amplitude => Eb = 1 (one sample per bit)

% ----- generate random bits and map to +/-1 (BPSK/NRZ)
b = randi([0 1], Nbits, 1);
s = 2*b - 1;                      % 0 -> -1, 1 -> +1

BER = zeros(size(EbN0));

for k = 1:numel(EbN0)
    % For BPSK real baseband: noise variance = N0/2 per real dimension.
    % With Eb=1, N0 = 1/EbN0; hence sigma^2 = 1/(2*EbN0).
    sigma2 = 1./(2*EbN0(k));
    n = sqrt(sigma2) * randn(Nbits,1);
    r = A*s + n;

    % hard decision
    bhat = r > 0;
    err  = nnz(bhat ~= b);
    BER(k) = err / Nbits;
    fprintf('Eb/N0 = %2d dB -> BER = %.3g (%d/%d)\n', EbN0dB(k), BER(k), err, Nbits);
end

% ----- theory (BPSK over AWGN)
BER_th = qfunc(sqrt(2*EbN0));

% ----- plot
figure('Name','Ex4.3.3 BER');
semilogy(EbN0dB, BER, 'o-','LineWidth',1.5); grid on; hold on;
semilogy(EbN0dB, BER_th, '--','LineWidth',1.5);
ylim([1e-6 1]); xlim([min(EbN0dB) max(EbN0dB)]);
xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('NRZ (BPSK) over AWGN: Simulated vs Theory');
legend('Simulated','Theoretical Q(\surd(2E_b/N_0))','Location','southwest');

% ----- explain / predict extremes in console
BER_min_shown = max(min(BER), 1e-6);
BER_max_shown = min(max(BER), 1.0);
fprintf('\nPlot y-range limited to [1e-6, 1]. In this run: min BER shown ≈ %.3g, max ≈ %.3g.\n', ...
        BER_min_shown, BER_max_shown);

BER_pred_lo = qfunc(sqrt(2*10^(-100/10)));   % SNR = -100 dB
BER_pred_hi = qfunc(sqrt(2*10^(+100/10)));   % SNR = +100 dB
fprintf('Predict at SNR = -100 dB: BER ≈ %.5f (≈ 0.5, random guesses).\n', BER_pred_lo);
fprintf('Predict at SNR = +100 dB: BER ≈ %.5g (≈ 0, error-free in practice).\n', BER_pred_hi);

end
