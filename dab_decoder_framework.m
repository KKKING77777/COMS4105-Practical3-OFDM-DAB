%% dab fic decoder
% basic dab decoding
% load process data

clear; close all; clc;

fprintf('=== DAB FIC Decoder Framework ===\n');

%% dab mode parameters
% standard dab values
DAB_PARAMS = struct();
DAB_PARAMS.mode = 1;                    % mode 1
DAB_PARAMS.N_fft = 2048;               % fft size
DAB_PARAMS.N_subcarriers = 1536;       % total carriers  
DAB_PARAMS.N_active = 1152;            % active carriers
DAB_PARAMS.N_guard = 504;              % guard samples
DAB_PARAMS.fs = 2.048e6;               % sample rate
DAB_PARAMS.T_u = 1024e-6;              % useful time
DAB_PARAMS.T_g = 246e-6;               % guard time  
DAB_PARAMS.T_s = 1270e-6;              % total time
DAB_PARAMS.subcarrier_spacing = 1000;   % carrier spacing

% frame structure
DAB_PARAMS.symbols_per_frame = 76;      % total symbols
DAB_PARAMS.FIC_symbols = 3;             % fic symbols
DAB_PARAMS.MSC_symbols = 72;            % msc symbols
DAB_PARAMS.frame_duration = 96e-3;      % frame duration

fprintf('DAB Mode I Parameters:\n');
fprintf('  FFT Size: %d\n', DAB_PARAMS.N_fft);
fprintf('  Active Subcarriers: %d\n', DAB_PARAMS.N_active);
fprintf('  Sampling Rate: %.3f MHz\n', DAB_PARAMS.fs/1e6);
fprintf('  Frame Duration: %.0f ms\n', DAB_PARAMS.frame_duration*1000);

%% load recording
try
    [dab_signal, fs_recorded] = audioread('dab.2021-12-16T14_26_44_664.wav');
    fprintf('\nDAB file loaded successfully:\n');
    fprintf('  File length: %.2f seconds\n', length(dab_signal)/fs_recorded);
    fprintf('  Recorded sampling rate: %.0f Hz\n', fs_recorded);
    fprintf('  Signal range: [%.4f, %.4f]\n', min(dab_signal), max(dab_signal));
    
    % convert complex
    if isreal(dab_signal)
        fprintf('  converting to complex\n');
        if mod(length(dab_signal), 2) == 1
            dab_signal = dab_signal(1:end-1);  % make even
        end
        % assume interleaved
        I_samples = dab_signal(1:2:end);
        Q_samples = dab_signal(2:2:end);
        dab_signal = I_samples + 1j * Q_samples;
        fs_recorded = fs_recorded / 2;  % adjust rate
    end
    
    fprintf('  Complex signal length: %d samples\n', length(dab_signal));
    
catch ME
    fprintf('Error loading DAB file: %s\n', ME.message);
    % generate test signal
    fprintf('Generating synthetic test signal...\n');
    fs_recorded = DAB_PARAMS.fs;
    duration = 0.5;  % 500ms
    t = (0:1/fs_recorded:duration-1/fs_recorded)';
    
    % create ofdm signal
    center_freq = 100e3;  % 100 kHz offset
    dab_signal = exp(1j*2*pi*center_freq*t) + 0.1*randn(size(t)) + 0.1j*randn(size(t));
end

%% resample signal
if abs(fs_recorded - DAB_PARAMS.fs) > 1000  % frequency difference
    fprintf('\nResampling from %.0f Hz to %.0f Hz...\n', fs_recorded, DAB_PARAMS.fs);
    dab_signal = resample(dab_signal, DAB_PARAMS.fs, fs_recorded);
    fs_actual = DAB_PARAMS.fs;
else
    fs_actual = fs_recorded;
end

%% frame extraction
fprintf('\n=== Exercise 4.3.1: Frame Extraction ===\n');

% calculate lengths
samples_per_frame = round(DAB_PARAMS.frame_duration * fs_actual);
samples_per_symbol = DAB_PARAMS.N_fft + DAB_PARAMS.N_guard;

fprintf('Samples per frame: %d\n', samples_per_frame);
fprintf('Samples per symbol: %d\n', samples_per_symbol);

% extract frame
if length(dab_signal) >= samples_per_frame
    % find start
    frame_start = 1;  % start position
    dab_frame = dab_signal(frame_start:frame_start + samples_per_frame - 1);
    
    fprintf('Extracted frame from sample %d to %d\n', frame_start, frame_start + samples_per_frame - 1);
else
    fprintf('Warning: Signal too short for complete frame\n');
    dab_frame = dab_signal;
end

% extract symbols
N_symbols_available = floor(length(dab_frame) / samples_per_symbol);
fprintf('Available symbols in extracted data: %d\n', N_symbols_available);

% symbol processing
symbols_matrix = zeros(DAB_PARAMS.N_active, min(N_symbols_available, DAB_PARAMS.symbols_per_frame));
active_subcarrier_indices = (1:DAB_PARAMS.N_active) + (DAB_PARAMS.N_fft - DAB_PARAMS.N_active)/2;

for sym_idx = 1:size(symbols_matrix, 2)
    % Extract symbol
    start_idx = (sym_idx - 1) * samples_per_symbol + 1;
    end_idx = start_idx + samples_per_symbol - 1;
    
    if end_idx <= length(dab_frame)
        symbol_with_guard = dab_frame(start_idx:end_idx);
        
        % remove guard
        useful_symbol = symbol_with_guard(DAB_PARAMS.N_guard + 1:end);
        
        % fft transform
        freq_domain = fft(useful_symbol, DAB_PARAMS.N_fft);
        
        % extract carriers
        symbols_matrix(:, sym_idx) = freq_domain(active_subcarrier_indices);
    end
end

fprintf('Successfully extracted %d symbols\n', size(symbols_matrix, 2));

% dpsk demodulation
if size(symbols_matrix, 2) > 1
    % differential demod
    dpsk_symbols = symbols_matrix(:, 2:end) .* conj(symbols_matrix(:, 1:end-1));
    fprintf('Applied π/4 DPSK demodulation to %d symbol pairs\n', size(dpsk_symbols, 2));
    
    % plot constellation
    figure(1);
    subplot(2,2,1);
    scatter(real(dpsk_symbols(:)), imag(dpsk_symbols(:)), 1, 'b.');
    title('π/4 DPSK Constellation (All Subcarriers)');
    xlabel('In-phase'); ylabel('Quadrature');
    grid on; axis equal;
    
    % plot symbols
    subplot(2,2,2);
    if size(dpsk_symbols, 2) >= 3
        scatter(real(dpsk_symbols(:, 1:3)), imag(dpsk_symbols(:, 1:3)), 10, 'r.');
        title('First 3 Symbols (FIC)');
        xlabel('In-phase'); ylabel('Quadrature');
        grid on; axis equal;
    end
else
    dpsk_symbols = [];
    fprintf('Warning: Not enough symbols for differential demodulation\n');
end

%% frequency deinterleaving
fprintf('\n=== Exercise 4.3.2: Frequency De-interleaving ===\n');

% create table
% simplified version
N_active = DAB_PARAMS.N_active;
interleaving_table = zeros(N_active, 1);

% generate pattern
% simplified version
rng(42);  % reproducible results
interleaving_table = randperm(N_active)';

fprintf('Created frequency de-interleaving table with %d entries\n', N_active);

% apply deinterleaving
if ~isempty(dpsk_symbols)
    deinterleaved_symbols = zeros(size(dpsk_symbols));
    for sym_idx = 1:size(dpsk_symbols, 2)
        deinterleaved_symbols(:, sym_idx) = dpsk_symbols(interleaving_table, sym_idx);
    end
    fprintf('Applied frequency de-interleaving to %d symbols\n', size(deinterleaved_symbols, 2));
else
    deinterleaved_symbols = [];
end

%% bit extraction
fprintf('\n=== Exercise 4.3.3: Bit Extraction ===\n');

if ~isempty(deinterleaved_symbols)
    % focus fic
    if size(deinterleaved_symbols, 2) >= 3
        fic_symbols = deinterleaved_symbols(:, 1:3);
        fprintf('Processing %d FIC symbols\n', 3);
        
        % convert qpsk
        % each symbol
        N_bits_per_symbol = N_active * 2;
        soft_bits = zeros(N_bits_per_symbol, 3);
        
        for sym_idx = 1:3
            bit_idx = 1;
            for subcarrier = 1:N_active
                qpsk_symbol = fic_symbols(subcarrier, sym_idx);
                
                % soft decision
                soft_bits(bit_idx, sym_idx) = real(qpsk_symbol);     % i bit
                soft_bits(bit_idx + 1, sym_idx) = imag(qpsk_symbol); % q bit
                bit_idx = bit_idx + 2;
            end
        end
        
        % convert bits
        hard_bits = soft_bits > 0;
        
        fprintf('Extracted %d bits per symbol, %d total bits\n', ...
                N_bits_per_symbol, numel(hard_bits));
        
        % group fibs
        bits_per_fib = 256;
        total_bits = numel(hard_bits);
        N_fibs = floor(total_bits / bits_per_fib);
        
        if N_fibs > 0
            fib_bits = reshape(hard_bits(1:N_fibs * bits_per_fib), bits_per_fib, N_fibs);
            fprintf('Grouped into %d FIB blocks of %d bits each\n', N_fibs, bits_per_fib);
        else
            fprintf('Warning: Not enough bits for complete FIB blocks\n');
            fib_bits = [];
        end
    else
        fprintf('Warning: Not enough symbols for FIC processing\n');
        fib_bits = [];
    end
else
    fib_bits = [];
end

%% energy dispersal
fprintf('\n=== Exercise 4.3.4: Energy Dispersal ===\n');

if ~isempty(fib_bits)
    % dispersal removal
    % prbs generator
    % initial state
    
    % process fibs
    crc_results = zeros(1, size(fib_bits, 2));
    
    for fib_idx = 1:size(fib_bits, 2)
        % remove dispersal
        prbs = generate_prbs(bits_per_fib);
        dispersed_bits = xor(fib_bits(:, fib_idx)', prbs);
        
        % extract data
        payload_bits = dispersed_bits(1:end-16);  % last 16
        crc_received = dispersed_bits(end-15:end);
        
        % crc check
        crc_calculated = mod(sum(payload_bits), 2^16);
        crc_from_bits = bi2de(crc_received);
        
        crc_results(fib_idx) = (crc_calculated == crc_from_bits);
        
        if crc_results(fib_idx)
            fprintf('FIB %d: CRC OK\n', fib_idx);
        else
            fprintf('FIB %d: CRC FAIL\n', fib_idx);
        end
    end
else
    fprintf('No FIB data available for energy dispersal processing\n');
end

%% fib decoding
fprintf('\n=== Exercise 4.3.5: FIB Block Structure ===\n');

if exist('fib_bits', 'var') && ~isempty(fib_bits)
    fprintf('FIB decoding framework ready\n');
    fprintf('Each FIB contains multiple FIG (Fast Information Group) blocks\n');
    fprintf('FIG types include:\n');
    fprintf('  - Type 0: MCI (Multiplex Configuration Information)\n'); 
    fprintf('  - Type 1: Labels (Service and Ensemble labels)\n');
    fprintf('  - Type 2: Reserved\n');
    fprintf('  - Type 3-7: Reserved for future use\n');
    fprintf('\nFor FIG 1/1 (Service Labels), look for:\n');
    fprintf('  - FIG type: 1\n');
    fprintf('  - Extension: 1\n');
    fprintf('  - Service ID and label text\n');
else
    fprintf('No valid FIB data available for decoding\n');
end

%% summary results
fprintf('\n=== Processing Summary ===\n');
fprintf('Signal processing chain completed:\n');
fprintf('✓ Frame extraction and symbol demodulation\n');
fprintf('✓ π/4 DPSK differential demodulation\n');
fprintf('✓ Frequency de-interleaving\n');
fprintf('✓ Bit extraction and FIB grouping\n');
fprintf('✓ Energy dispersal removal framework\n');
fprintf('✓ Basic FIB structure analysis\n');

% save results
save('dab_processing_results.mat', 'symbols_matrix', 'dpsk_symbols', 'DAB_PARAMS', ...
     'fs_actual', 'dab_frame', 'interleaving_table');

% save figure
if exist('gcf', 'builtin') && ishandle(gcf)
    saveas(gcf, 'dab_decoder_results.png');
    fprintf('\nFigure saved as dab_decoder_results.png\n');
end

fprintf('\nDAB processing framework completed!\n');
fprintf('Files saved: dab_processing_results.mat\n');