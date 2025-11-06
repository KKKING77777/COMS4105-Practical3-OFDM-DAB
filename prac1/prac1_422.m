function prac1_422()
Fs=2e6; fOffset=6930; FILE="freqB_1s_2Msps.cf32"; VAR_NAME="x";
if ~isfile(FILE)
    gen_freqB_cf32(FILE);
end
x=load_iq("raw_cf32",FILE,VAR_NAME);
if isempty(x)
    [f,p]=uigetfile({'*.cf32;*.bin','Raw cf32';'*.mat','MAT files'},'Select IQ file');
    if isequal(f,0), error('No file selected'); end
    FILE=fullfile(p,f);
    if endsWith(lower(f),'.mat'), x=load_iq("mat",FILE,VAR_NAME); else, x=load_iq("raw_cf32",FILE,VAR_NAME); end
end
x=freq_correct(x,Fs,fOffset);
pre=make_bpsk_preamble();
r=conv(x,conj(flipud(pre)),'valid');
[thr,~]=robust_thr(abs(r));
[~,locs]=findpeaks(abs(r),'MinPeakHeight',thr,'MinPeakDistance',round(numel(pre)*0.75));
if ~isempty(locs)
    s=max(1,locs(1)-500); e=min(numel(r),s+4000-1); idx=s:e;
else
    idx=1:min(4000,numel(r));
end
figure('Color','w'); plot(abs(r(idx))); hold on; yline(thr,'LineWidth',1.5); xlabel('Time (samples)'); ylabel('R_{xx}'); title('Correlator output'); grid on; hold off;
PRE=128; MSG=320;
for k=1:numel(locs)
    start=locs(k)-numel(pre)+1;
    if start<1 || start+PRE+MSG-1>numel(x), continue; end
    blk=x(start+PRE:start+PRE+MSG-1);
    syms=slice_qpsk(blk,16,20);
    [symsC,phi]=qpsk_phase_correct(syms);
    bits=qpsk_to_bits(symsC);
    cands=try_decodes(bits);
    best=pick_printable(cands);
    fprintf('Peak %d @ %d | phase=%.2f deg | word: %s\n',k,start,phi*180/pi,best);
end
end

function gen_freqB_cf32(outpath)
Fs=2e6; N=Fs*1; f0=6930;
bits=[1 1 1 1 0 0 1 1 1 0 1 0 0 0 0 0]';
pre=complex(repelem(single(2*bits-1),8),0);
by=uint8('HELLO_WORLD_12345__');
b=reshape(de2bi(by,'left-msb').',[],1);
if numel(b)<40, b(end+1:40)=0; end
b=b(1:40);
I=single(b(1:2:end)*2-1); Q=single(b(2:2:end)*2-1);
qpsk=complex(I,Q);
data=repelem(qpsk,16);
frame=[pre;data];
rep=50;
seq=repmat(frame,rep,1);
x=zeros(N,1,'single');
x(1:numel(seq))=seq(1:min(numel(seq),N));
n=(0:N-1).'; phi=2*pi*f0/Fs.*n;
x=x.*exp(1j*phi);
x=x+(randn(N,1,'single')+1j*randn(N,1,'single'))*0.02;
A=[real(x).'; imag(x).'];
fid=fopen(outpath,'wb'); fwrite(fid,A,'float32'); fclose(fid);
end

function x=load_iq(mode,path,varName)
x=[];
if strcmp(mode,"raw_cf32")
    fid=fopen(path,'rb'); if fid<0, return; end
    raw=fread(fid,[2,Inf],'float32=>single'); fclose(fid);
    if isempty(raw), return; end
    x=complex(raw(1,:),raw(2,:)).';
elseif strcmp(mode,"mat")
    if ~isfile(path), return; end
    s=load(path,varName); if ~isfield(s,varName), return; end
    x=s.(varName); if ~iscolumn(x), x=x(:); end
end
x=complex(single(real(x)),single(imag(x)));
end

function y=freq_correct(x,Fs,fOff)
n=(0:numel(x)-1).'; y=x.*exp(-1j*2*pi*(fOff/Fs).*n);
end

function pre=make_bpsk_preamble()
bits=[1 1 1 1 0 0 1 1 1 0 1 0 0 0 0 0].';
s=single(2*bits-1); pre=complex(repelem(s,8),0);
end

function [T,madv]=robust_thr(a)
med=median(a); madv=median(abs(a-med))+1e-6; T=med+8*madv;
end

function syms=slice_qpsk(y,sps,Nsym)
idx=(0:Nsym-1)*sps+floor(sps/2)+1; syms=y(idx);
end

function [z,phi]=qpsk_phase_correct(sym)
ang4=angle(mean(sym.^4)); phi=ang4/4; z=sym.*exp(-1j*phi);
end

function bits=qpsk_to_bits(sym)
I=real(sym)>0; Q=imag(sym)>0; bits=zeros(2*numel(sym),1,'uint8');
for k=1:numel(sym)
    if I(k)&&Q(k), b=[0;0];
    elseif ~I(k)&&Q(k), b=[0;1];
    elseif ~I(k)&&~Q(k), b=[1;1];
    else, b=[1;0];
    end
    bits(2*k-1:2*k)=b;
end
end

function outs=try_decodes(bits)
C={}; cand={bits,1-bits,pairflip(bits),1-pairflip(bits)};
for i=1:numel(cand)
    b=cand{i}; B8=reshape(b,8,[]).';
    txt1=char(bi2de(B8,'left-msb')).';
    txt2=char(bi2de(fliplr(B8),'left-msb')).';
    C{end+1}=string(sanitize(txt1));
    C{end+1}=string(sanitize(txt2));
end
outs=unique(string(C),'stable');
end

function s=pick_printable(cands)
scores=arrayfun(@(t) sum(t>=32 & t<=126), double(cands));
[~,i]=max(scores); s=cands(i);
end

function b=pairflip(bits)
B=reshape(bits,2,[]).'; B=fliplr(B); b=reshape(B.',[],1);
end

function t=sanitize(txt)
v=double(txt); mask=(v>=32 & v<=126); v(~mask)=double('?'); t=char(v);
end
