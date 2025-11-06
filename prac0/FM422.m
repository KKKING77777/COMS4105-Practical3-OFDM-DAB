clear; clc;

% ==== 参数设置 ====
filename = 'fm.bin';   
fs = 250e3;            % 采样率 (FM 250 kS/s, DAB 2e6)
fc = 100.3e6;          % 中心频率 (例如 100.3 MHz)

% ==== 读取 IQ 数据 ====
fid = fopen(filename,'rb');
if fid < 0
    error('cannot open', filename);
end
raw = fread(fid, inf, 'uint8=>single'); fclose(fid);
raw = (raw - 127.5) / 128.0;   % 转换到 -1~+1
I = raw(1:2:end); 
Q = raw(2:2:end);
x = complex(I, Q);

% ==== FFT ====
N = length(x);
X = fftshift(fft(x, N));          
Pxx = abs(X).^2 / N;              
Pxx_dB = 10*log10(Pxx + eps);     
% ==== 频率轴 ====
f = (-N/2:N/2-1)*(fs/N);          

% ==== 绘图 ====
plot(f/1e3, Pxx_dB);  
xlabel('Frequency offset (kHz)');
ylabel('Power (dB)');
title(['Spectrum around ', num2str(fc/1e6), ' MHz']);
grid on;
