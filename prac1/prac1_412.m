function prac1_412
% 4.1.2 AWGN（BPSK）—目标误码数法
clc; rng(1);

EbN0dB    = [0 2 4 6 8 10 12];
ErrTarget = 50;          % 每个点至少统计到的误码个数
MaxBits   = 5e7;         % 每个点的最大比特数上限，防止极端耗时
Chunk     = 2e5;         % 分批仿真块

res = zeros(numel(EbN0dB),7);
for k = 1:numel(EbN0dB)
    err = 0; N = 0;
    ebn0 = 10^(EbN0dB(k)/10);     % 线性值
    % BPSK基带：设符号能量=比特能量=1 => N0 = 1/ebn0，噪声方差 = N0/2
    sigma = sqrt(1/(2*ebn0));

    while err < ErrTarget && N < MaxBits
        b  = randi([0 1], Chunk, 1, 'uint8');
        s  = 1 - 2*double(b);                 % ±1
        r  = s + sigma*randn(Chunk,1);        % AWGN
        bh = r < 0;
        err = err + sum(b~=bh);
        N   = N + Chunk;
    end

    p   = err/max(N,1);
    ci  = 1.96*sqrt(max(p*(1-p)/N, eps));     % 95% 近似置信区间
    pth = 0.5*erfc(sqrt(ebn0));               % 理论BER: Q(sqrt(2Eb/N0)) = 0.5*erfc(sqrt(Eb/N0))

    res(k,:) = [EbN0dB(k) N err p pth max(p-ci,0) min(p+ci,1)];
end

T = array2table(res, 'VariableNames', ...
 {'EbN0_dB','Bits_Used','Errors','BER_Sim','BER_Theory','BER_Lower95','BER_Upper95'});

disp('=== 4.1.2 AWGN（BPSK）—目标误码数法 ===');
disp(T);

% 画图（可选）
figure; semilogy(T.EbN0_dB, T.BER_Sim, 'o-'); hold on;
semilogy(T.EbN0_dB, T.BER_Theory, '--'); grid on; hold off;
xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('BPSK over AWGN'); legend('Simulation','Theory (0.5·erfc(\surd(E_b/N_0)))','Location','southwest');
end
