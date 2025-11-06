function prac1_423()
Fs=2e6; fOff=9832.2; SPSb=8; SPSq=16;

pre_bits  = [1 1 1 1 0 0 1 1 1 0 1 0 0 0 0 0];
train_bits= [0 0 0 0  1 1 1 1  0 0 1 1  1 1 0 0  1 0 1 0  0 1 0 1  1 1 1 0  0 1 1 1];
msg='EQUALISER_16QAM_TEST';

x = make_pkt(pre_bits,train_bits,msg,Fs,SPSb,SPSq,fOff);

pre = complex(repelem(single(2*pre_bits-1),SPSb),0);
x   = freq_correct(x,Fs,fOff);
r   = conv(x,conj(flipud(pre)),'valid');
[thr,~]=robust_thr(abs(r));
[~,locs]=findpeaks(abs(r),'MinPeakHeight',thr,'MinPeakDistance',round(numel(pre)*0.75));
if isempty(locs), error('no preamble'); end
start = locs(1)-numel(pre)+1;

tr   = bits_to_16qam(train_bits);
best = struct('metric',-inf,'off',0,'a',1+0j,'rs',[],'seg',[],'centers',[]);
for off = -6:6
    centers = (0:numel(tr)-1)*SPSq + (floor(SPSq/2)+1+off);
    seg = x(start+numel(pre) : start+numel(pre)+centers(end)-1);
    rs  = take_avg(seg, centers, 1);       % 小窗平均采样
    a   = ls_scalar(rs, tr);
    metr = abs(sum(conj(tr).*(rs./a)));    % 等化后相关幅度
    if metr > best.metric
        best.metric = metr; best.off = off; best.a = a;
        best.rs = rs; best.seg = seg; best.centers = centers;
    end
end

% 一次性剔除异常点后再估计
rs0 = best.rs; a0 = best.a;
rs_eq0 = rs0./a0;
ok = abs(rs_eq0) < 6;                       % 阈值内视为正常
a = ls_scalar(rs0(ok), tr(ok));
rs_eq = rs0./a;

seg_eq = best.seg./a;

figure('Color','w');
subplot(2,1,1); plot(real(rs_eq),'r.-'); hold on; plot(real(tr),'b.-'); grid on; title('I component (equalised vs ideal)');
subplot(2,1,2); plot(imag(rs_eq),'r.-'); hold on; plot(imag(tr),'b.-'); grid on; title('Q component (equalised vs ideal)');

cent_all = (0:numel(tr)+80-1)*SPSq + (floor(SPSq/2)+1+best.off);
seg2 = x(start+numel(pre) : start+numel(pre)+cent_all(end)-1);
data_syms = take_avg(seg2, cent_all, 1)./a;
data_syms = data_syms(numel(tr)+1:end);

idx  = map_16qam_to_idx(data_syms);
disp('Best timing offset (samples):'), disp(best.off)
disp('First DATA symbol indices:'),   disp(idx(1:min(12,end))')
bits = idx_to_bits(idx);
txts = try_pack_ascii(bits);
disp('ASCII candidates:'),            disp(txts(1:min(3,end)))
end

% ===== helpers =====
function x = make_pkt(pre_bits,train_bits,msg,Fs,SPSb,SPSq,f0)
pre = complex(repelem(single(2*pre_bits-1),SPSb),0); pre=pre(:);
tr  = bits_to_16qam(train_bits); tr_up = repelem(tr(:),SPSq,1);
db  = reshape(de2bi(uint8(msg),'left-msb').',[],1).';
pad = mod(-numel(db),4); if pad<0, pad=pad+4; end
db  = [db zeros(1,pad)];
ds  = bits_to_16qam(db); ds_up = repelem(ds(:),SPSq,1);
w   = [pre; tr_up; ds_up];
N   = Fs*1; x=zeros(N,1,'single'); L=min(N,numel(w)); x(1:L)=w(1:L);
n=(0:N-1).'; x=x.*exp(1j*2*pi*f0/Fs.*n);
x=x+(randn(N,1,'single')+1j*randn(N,1,'single'))*0.01;
end

function y=freq_correct(x,Fs,fOff)
n=(0:numel(x)-1).'; y=x.*exp(-1j*2*pi*(fOff/Fs).*n);
end

function s=bits_to_16qam(b)
B=reshape(b,4,[]).';
gI=bi2de(B(:,1:2),'left-msb'); gQ=bi2de(B(:,3:4),'left-msb');
lev=[-3 -1 3 1]; I=single(lev(gI+1)); Q=single(lev(gQ+1));
s=complex(I,Q); s=s(:);
end

function idx=map_16qam_to_idx(s)
I=real(s(:)); Q=imag(s(:)); lev=[-3 -1 1 3];
[~,iI]=min(abs(I-lev.'),[],1); [~,iQ]=min(abs(Q-lev.'),[],1);
idx=((iI-1)*4+(iQ-1)).';
end

function b=idx_to_bits(idx)
g=idx(:); gi=floor(g/4); gq=mod(g,4);
Ibits=de2bi(gi,2,'left-msb'); Qbits=de2bi(gq,2,'left-msb');
B=[Ibits Qbits]; b=reshape(B.',[],1).';
end

function t=try_pack_ascii(b)
pad=mod(-numel(b),8); if pad<0, pad=pad+8; end
bb=[b zeros(1,pad)];
B8=reshape(bb,8,[]).';
txt1=char(bi2de(B8,'left-msb')).';
txt2=char(bi2de(fliplr(B8),'left-msb')).';
t=unique(string({txt1,txt2}),'stable');
end

function [T,madv]=robust_thr(a)
med=median(a); madv=median(abs(a-med))+1e-6; T=med+8*madv;
end

function rs = take_avg(seg, centers, rad)
K=numel(centers); rs=zeros(K,1,'like',seg);
for k=1:K
    c=centers(k);
    s=max(1,c-rad); e=min(numel(seg),c+rad);
    rs(k)=mean(seg(s:e));
end
end

function a = ls_scalar(rs, tr)
num = sum(conj(tr).*rs);
den = sum(conj(tr).*tr) + 1e-12;
a = num/den;
end
