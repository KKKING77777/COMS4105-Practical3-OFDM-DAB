function prac1_43
% Timing recovery demo (optimized, single file)

    close all; clc;
    if exist('recording.mat','file'), S = load('recording.mat'); x=S.x; fs=S.fs;
    else, [x,fs]=synth_bursts(); end

    segs = energy_gate(x,256,-3,5000);
    figure('Name','Integrate-and-Dump Constellation'); hold on; axis equal; grid on;
    xlabel('I'); ylabel('Q'); title('Integrate-and-Dump Constellation');

    for k=1:size(segs,1)
        pkt = x(segs(k,1):segs(k,2));

        % coarse CFO + power norm
        [pkt,~] = coarse_freq_correct(pkt,fs);
        pkt = pkt ./ sqrt(mean(abs(pkt).^2)+eps);

        % RRC MF + integer sps (narrow search)
        [sps,y_mf] = rrc_and_estimate_sps(pkt,6,12,0.35,8);

        % Gardner timing (fractional, cubic interp, stable)
        y = gardner_timing(y_mf,sps,0.008,0.0);   % mu_ted, mu_omega

        % fine CFO by linear fit on angle(y.^4)
        y = fine_cfo_linear4(y);

        % Costas loop (QPSK)
        y = costas_qpsk(y,0.006,0.15);

        % 1-tap DD-LMS equalizer
        y = dd_lms_onetap(y,0.010,2);

        % keep data-like symbols only
        amp = abs(y);
        y = y(amp > max(0.55,0.80*median(amp)));

        % AGC + phase centering
        y = y ./ sqrt(mean(abs(y).^2)+eps);
        y = y .* exp(-1i*angle(mean(slicer_qpsk(y))));

        % EVM + light outlier reject (plot only)
        [evmPct,y_id] = evm_qpsk(y);
        keep = abs(y) <= prctile(abs(y),99.5);
        plot(real(y(keep)),imag(y(keep)),'.');
        plot(real(y_id(keep)),imag(y_id(keep)),'k+','MarkerSize',3);

        fprintf('Packet %d: sps=%d, Ns=%d, EVM=%.1f%%\n',k,sps,numel(y),evmPct);
    end
    hold off;

    fprintf('\n== 4.3.2 OFDM/DAB symbol-length demo ==\n');
    demo_4_3_2();
end

% ------------ timing / CFO blocks ------------

function [sps_best,y_mf]=rrc_and_estimate_sps(x,sps_min,sps_max,beta,span)
% RRC MF per candidate; choose best integer sps
    Jbest=-inf; sps_best=sps_min;
    for sps=sps_min:sps_max
        h=rrc(beta,span,sps);
        y=conv(x,conj(flipud(h)),'same');
        gd=(length(h)-1)/2; i0=round(gd)+1;
        if i0>length(y),continue;end
        y_dec=y(i0:sps:end); if numel(y_dec)<50,continue;end
        J=mean(abs(y_dec));
        if J>Jbest, Jbest=J; sps_best=sps; end
    end
    h=rrc(beta,span,sps_best);
    y_mf=conv(x,conj(flipud(h)),'same');
end

function y_sym=gardner_timing(y_mf,sps,mu_ted,mu_omega)
% Gardner TED with cubic interpolation
    N=numel(y_mf); nvec=(1:N).';
    t=max(sps*2,floor(N/10)); omega=sps; k=1;
    y_sym=complex(zeros(floor(N/sps),1));
    while t+omega+2<=N
        xp=interp1(nvec,y_mf,t-omega, 'pchip','extrap');
        xm=interp1(nvec,y_mf,t-omega/2,'pchip','extrap');
        xc=interp1(nvec,y_mf,t,       'pchip','extrap');
        e=real((xp-xc)*conj(xm));
        omega=omega+mu_omega*e;
        t=t+omega+mu_ted*e;
        y_sym(k)=xc; k=k+1;
        if k>numel(y_sym), y_sym(end+1000,1)=0; end %#ok<AGROW>
    end
    y_sym=y_sym(1:k-1);
end

function z=fine_cfo_linear4(y)
% Linear fit on angle(y.^4)
    n=numel(y); if n<16, z=y; return; end
    a4=unwrap(angle(y.^4)); k=(0:n-1).';
    p=polyfit(k,a4,1); w4=p(1);
    z=y.*exp(-1i*(w4/4)*k);
    c=angle(mean(z.^4))/4; z=z*exp(-1i*c);
end

function y=costas_qpsk(x,mu,alpha)
% PI Costas with soft decisions
    theta=0; v=0; y=zeros(size(x));
    for n=1:numel(x)
        z=x(n)*exp(-1i*theta); y(n)=z;
        sI=tanh(3*real(z)); sQ=tanh(3*imag(z));
        e=sI*imag(z)-sQ*real(z);
        v=v+mu*e; theta=theta+alpha*e+v;
    end
end

function y=dd_lms_onetap(x,mu,passes)
% Decision-directed one-tap LMS
    a=1+0i; y=zeros(size(x));
    for p=1:passes
        for n=1:numel(x)
            y(n)=a*x(n); d=slicer_qpsk(y(n)); e=d-y(n);
            a=a+mu*e*conj(x(n));
        end
    end
end

% ------------ signal model / utils ------------

function [x,fs]=synth_bursts
% QPSK bursts for offline test
    rng(1); fs=1e6; spsList=[8 6 10];
    x=complex([]); gap=zeros(round(fs/50),1);
    for sps=spsList
        bits=randi([0 1],1600,1);
        syms=qpsk_mod(bits);
        bb=upsample_filter_rrc(syms,sps,0.35,8);
        x=[x;bb;gap]; %#ok<AGROW>
    end
    x=awgnComplex(x,24);
    fo=1200; n=(0:numel(x)-1).';
    x=x.*exp(1i*2*pi*fo*n/fs)*exp(1i*pi/6);
end

function h=rrc(beta,span,sps)
% Root-raised-cosine pulse
    t=(-span*sps:span*sps).' / sps; h=zeros(size(t));
    for i=1:numel(t)
        ti=t(i);
        if abs(ti)<1e-12
            h(i)=1-beta+4*beta/pi;
        elseif abs(abs(ti)-1/(4*beta))<1e-12
            h(i)=(beta/sqrt(2))*((1+2/pi)*sin(pi/(4*beta))+(1-2/pi)*cos(pi/(4*beta)));
        else
            h(i)=(sin(pi*ti*(1-beta))+4*beta*ti*cos(pi*ti*(1+beta)))/(pi*ti*(1-(4*beta*ti)^2));
        end
    end
    h=h/sqrt(sum(h.^2));
end

function y=upsample_filter_rrc(sym,sps,beta,span)
    x=upsample(sym,sps); h=rrc(beta,span,sps);
    y=conv(x,h,'same');
end

function s=qpsk_mod(bits)
    b=reshape(bits,[],2);
    s=((2*b(:,1)-1)+1i*(2*b(:,2)-1))/sqrt(2);
end

function y=awgnComplex(x,snrdb)
    p=mean(abs(x).^2); nvar=p/10^(snrdb/10);
    y=x+sqrt(nvar/2)*(randn(size(x))+1i*randn(size(x)));
end

function e=moving_rms(x,w)
    e=sqrt(conv(abs(x).^2,ones(w,1)/w,'same'));
end

function segs=energy_gate(x,win,thr_db,min_gap)
% Burst finder
    e=moving_rms(x,win); thr=median(e)*10^(thr_db/20);
    idx=find(e>thr); if isempty(idx), segs=[]; return; end
    s=idx(1); p=idx(1); segs=[];
    for i=2:numel(idx)
        if idx(i)-p>min_gap
            segs=[segs; max(1,s-win) min(numel(x),p+win)]; %#ok<AGROW>
            s=idx(i);
        end; p=idx(i);
    end
    segs=[segs; max(1,s-win) min(numel(x),p+win)];
end

function [xc,fhat]=coarse_freq_correct(x,fs)
% Mean diff-phase CFO
    z=x(2:end).*conj(x(1:end-1));
    w=angle(mean(z)); fhat=w*fs/(2*pi);
    n=(0:numel(x)-1).'; xc=x.*exp(-1i*2*pi*fhat*n/fs);
end

function [evmPct,y_ideal]=evm_qpsk(y)
% EVM vs ideal QPSK
    c=(1/sqrt(2))*[1+1i;1-1i;-1+1i;-1-1i];
    y_ideal=zeros(size(y));
    for n=1:numel(y), [~,ix]=min(abs(y(n)-c)); y_ideal(n)=c(ix); end
    evmPct=100*sqrt(mean(abs(y-y_ideal).^2)/mean(abs(y_ideal).^2));
end

function s=slicer_qpsk(y)
    s=(1/sqrt(2))*(sign(real(y))+1i*sign(imag(y)));
end

% ------------ 4.3.2 demo ------------

function L=estimate_symbol_len_generic(x,search12)
    x0=x-mean(x); r=xcorr(x0,'biased'); r=r(numel(x0):end);
    a=search12(1); b=search12(2); [~,k]=max(real(r(a:b))); L=a+k-1;
end

function demo_4_3_2
    rng(2); N=128; CP=32; Ns=N+CP; M=50;
    X=(randn(M,N)+1i*randn(M,N))/sqrt(2);
    x=ifft(X,[],2); x=reshape([x(:,end-CP+1:end) x].',[],1);
    x=awgnComplex(x,20);
    L=estimate_symbol_len_generic(x,[120 200]);
    fprintf('Estimated OFDM symbol length ≈ %d (true=%d)\n',L,Ns);
end
