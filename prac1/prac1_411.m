function prac1_411
% 4.1.1 无噪声（BPSK）—无工具箱版本
clc; rng(1);

Nbits = 1e6;          % 比特数
Ns    = 8;            % 每符号采样点
Ts    = 1;            % 符号时长(任意单位)
Fs    = Ns/Ts;        % 采样率
fc    = Fs/4;         % 载波(避免混叠)

% 生成比特并BPSK映射：0->+1, 1->-1
b = randi([0 1], Nbits, 1, 'uint8');
s = 1 - 2*double(b);               % ±1

% 上采样
x = repelem(s, Ns);

% 上变频到载波
n  = (0:numel(x)-1).';
tx = x .* exp(1j*2*pi*fc/Fs*n);

% 无噪声信道
rx = tx;

% 下变频到基带
bb = rx .* exp(-1j*2*pi*fc/Fs*n);

% 区间积分取样（匹配滤波等效）
y  = intdump(bb, Ns)/Ns;

% 判决
bh = real(y) < 0;
BER = mean(b ~= bh);

fprintf('=== 4.1.1 无噪声（BPSK）===\n比特数 = %d\nBER = %.6f\n', Nbits, BER);
end

function y = intdump(x, Ns)
% 简易 integrate-and-dump
L = floor(numel(x)/Ns)*Ns;
x = reshape(x(1:L), Ns, []);
y = sum(x,1).';
end
