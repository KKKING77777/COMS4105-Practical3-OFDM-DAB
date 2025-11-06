function prac1_424()
Fs=2e6; Tu=2000; Tg=492; Ts=Tu+Tg;
x=gen_dab_like(Fs,Tu,Tg,Fs);
w=round(0.5*Tg);
ener=movmean(abs(x).^2,w);
[~,null_end]=min(ener(1:round(0.15*numel(x))));
seg=x(max(1,null_end+1):min(numel(x),null_end+6*Ts));
L=length(seg); Npos=L-(Tu+Tg-1);
P=zeros(Npos,1,'single');
for d=1:Npos
    a=seg(d:d+Tg-1);
    b=seg(d+Tu:d+Tu+Tg-1);
    P(d)=sum(a.*conj(b));
end
M=movmean(abs(P).^2,32);
med=median(M); madv=median(abs(M-med))+1e-12; thr=med+6*madv;
[~,locs]=findpeaks(double(M),'MinPeakHeight',thr,'MinPeakDistance',round(0.8*Ts));
if isempty(locs), error('no peaks'); end
frame_start=null_end+locs(1);
period=round(mean(diff(locs)));
cfo=angle(P(locs(1)))/(2*pi*(Tu/Fs));
n=(0:numel(x)-1).';
x_c=x.*exp(-1j*2*pi*cfo*n/Fs);
disp(['frame_start = ',num2str(frame_start)])
disp(['CFO_Hz      = ',num2str(cfo)])
disp(['peak period ≈ ',num2str(period),' / ',num2str(Ts)])
figure('Color','w');
subplot(2,1,1); plot(ener); title('Energy (null detection)'); grid on
subplot(2,1,2); plot(M); hold on; yline(thr,'k-'); plot(locs,M(locs),'ro'); title('CP metric with peaks'); grid on
blk=x_c(frame_start:frame_start+Ts-1);
sym=blk(Tg+1:end);
X=fft(sym);
figure('Color','w'); plot(real(X),imag(X),'.'); axis equal; title('Constellation (1st OFDM symbol)')
end

function x=gen_dab_like(Fs,Tu,Tg,N)
x=zeros(N,1,'single'); nullL=round(0.02*N); x(1:nullL)=0; ptr=nullL+1; reps=40; rng(1);
for r=1:reps
    S=(randn(Tu,1)+1j*randn(Tu,1));
    s=ifft(S);
    ofdm=[s(end-Tg+1:end); s];
    L=min(length(ofdm),N-ptr+1);
    x(ptr:ptr+L-1)=ofdm(1:L);
    ptr=ptr+L; if ptr>N, break; end
end
x=x+(randn(N,1,'single')+1j*randn(N,1,'single'))*0.01;
f0=9832; n=(0:N-1).'; x=x.*exp(1j*2*pi*f0*n/Fs);
end
