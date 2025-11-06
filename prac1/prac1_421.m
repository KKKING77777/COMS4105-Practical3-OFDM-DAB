function prac1_421
clc; close all;

% ===== 参数 =====
sps = 16;                 
SNR_dB = 22;              
fs_in = [];               
start_hint = [];          

% ===== 读取/合成 =====
[iq, fs] = load_or_synth(fs_in, sps, SNR_dB);
iq = iq(:);

% ===== 能量起点 =====
env = movmean(abs(iq), 200);
thr = median(env) + 4*mad0(env);
if isempty(start_hint)
    k = find(env > thr, 1, 'first'); if isempty(k), k = 1; end
    start_idx = max(1, k - 10*sps);
else
    start_idx = max(1, start_hint);
end
x = iq(start_idx:end);

% ===== 频偏估计（鲁棒版）=====
% 粗估：去调制 z = x.^4，1点延迟相位差 → f_coarse
z = x.^4;
phi = angle(z(2:end).*conj(z(1:end-1)));
f_coarse = (fs/(2*pi*4)) * mean(phi);

% 细扫：在粗估 ±3 kHz 内细搜索（2 Hz 步进）
n = (0:numel(x)-1).';
span = 3000; step = 2;
ff = (f_coarse-span):step:(f_coarse+span);
score = zeros(size(ff));
for i = 1:numel(ff)
    w = exp(-1j*2*pi*ff(i)/fs*n);
    y4 = (x.*w).^4;
    score(i) = abs(mean(y4));
end
[~, ix] = max(score);
f0 = ff(ix);

% 频偏校正
xc = x .* exp(-1j*2*pi*f0/fs*n);

% ===== 相位对齐 + 积分抽取（搜索0..sps-1最佳相位）=====
[y, k_best] = best_intdump(xc, sps);

% ===== 画图 =====
figure(1); clf;
plot(env,'b'); hold on; xline(start_idx,'r--','start','LabelOrientation','aligned');
title('Energy |x|'); xlabel('sample'); grid on;

figure(2); clf;
scatter(real(y), imag(y), 8, '.'); axis equal; grid on;
title(sprintf('Constellation after sync (best phase = %d, CFO=%.1f Hz)', k_best, f0));
xlabel('I'); ylabel('Q');

if exist('eyediagram','file')==2
    figure; eyediagram(real(xc), 2*sps); title('Eye (2 symbols)');
end
end

% ===== 辅助 =====
function [iq, fs] = load_or_synth(fs_in, sps, SNR_dB)
if evalin('base','exist(''iq'',''var'')')
    iq = evalin('base','iq'); iq = iq(:);
    if ~isempty(fs_in), fs = fs_in; else
        if evalin('base','exist(''fs'',''var'')'), fs = evalin('base','fs'); else, fs = 2e6; end
    end
    return;
end
[iq, fs] = try_files(fs_in);
if ~isempty(iq), iq = iq(:); if isempty(fs), fs=2e6; end, return; end
if isempty(fs_in), fs=2e6; else, fs=fs_in; end
M = 4; const = (1/sqrt(2))*[1+1j; -1+1j; -1-1j; 1-1j];
rng(1);
pre_len = 256; Ns = 10000; silence = 6645;
s_all = [const(randi([1 M],pre_len,1)); const(randi([1 M],Ns,1))];
tx = upsample(s_all,sps); tx = filter(ones(sps,1),1,tx);
tx = [zeros(silence,1); tx];
tx = awgn0(tx, SNR_dB);
n = (0:numel(tx)-1).';
f0_syn = -1248;                        
iq = tx .* exp(1j*(2*pi*f0_syn/fs*n + pi/6));
end

function [iq, fs] = try_files(fs_in)
iq=[]; fs=[];
if exist('iq.mat','file')
    S=load('iq.mat'); if isfield(S,'iq'), iq=S.iq; end
    if isempty(fs_in) && isfield(S,'fs'), fs=S.fs; else, fs=fs_in; end
    if ~isempty(iq), return; end
end
if exist('iq.bin','file')
    fid=fopen('iq.bin','rb'); v=fread(fid,'float32'); fclose(fid);
    if mod(numel(v),2)==0, v=reshape(v,2,[]).'; iq=complex(v(:,1),v(:,2)); fs=fs_in; end
    if ~isempty(iq), if isempty(fs), fs=2e6; end, return; end
end
if exist('iq.csv','file')
    M=readmatrix('iq.csv'); if size(M,2)>=2, iq=complex(M(:,1),M(:,2)); fs=fs_in; if isempty(fs), fs=2e6; end, return; end
end
end

function [y_best, k_best] = best_intdump(x, sps)
best = -inf; y_best=[]; k_best=0;
for k=0:sps-1
    yk = intdump0(x(k+1:end), sps);
    sc = -var(angle(yk.^4));
    if sc>best, best=sc; y_best=yk; k_best=k; end
end
end

function y = intdump0(x,M)
if exist('intdump','file')==2
    L=floor(numel(x)/M)*M; y=intdump(x(1:L),M);
else
    L=floor(numel(x)/M)*M; xx=reshape(x(1:L),M,[]); y=sum(xx,1).';
end
end

function m = mad0(x)
med = median(x); m = median(abs(x-med));
end

function y = awgn0(x,SNRdB)
Px=mean(abs(x).^2); Np=Px/10^(SNRdB/10);
y = x + sqrt(Np/2)*(randn(size(x))+1j*randn(size(x)));
end
