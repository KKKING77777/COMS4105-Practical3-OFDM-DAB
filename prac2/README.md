# COMS4105 Practical 2 - Channel Coding and Frequency Synchronisation

Complete implementation for COMS4105 Practical 2 covering channel coding methods and over-the-air signal processing.

## Files Structure

**Preparation Questions:**
- `preparation_answers.md` - Complete answers to all 5 preparation questions

**Exercise Implementations:**
- `exercise_4_1_1.m` - Hamming (15,11) encoder/decoder with BER testing
- `exercise_4_1_2.m` - CRC-16 error detection system  
- `exercise_4_1_3.m` - Convolutional encoder/decoder with Viterbi algorithm
- `exercise_4_1_4.m` - Reed-Solomon (255,239) encoder/decoder
- `exercise_4_2_1.m` - Frequency synchronization with over-the-air signals
- `exercise_4_2_2.m` - Hamming-decoded text message extraction
- `exercise_4_2_3.m` - Hamming + CRC-16 frame decoding
- `exercise_4_3_1.m` - Convolutional decoder for frame type 0001
- `exercise_4_3_2.m` - Reed-Solomon decoder for frame type 0010  
- `exercise_4_3_3.m` - Concatenated RS+Convolutional with 16-QAM

## Running the Code

Execute exercises in order:

```matlab
% Channel coding performance comparison
run('exercise_4_1_1.m')  % Hamming codes
run('exercise_4_1_2.m')  % CRC-16 detection
run('exercise_4_1_3.m')  % Convolutional codes
run('exercise_4_1_4.m')  % Reed-Solomon codes

% Over-the-air signal processing
run('exercise_4_2_1.m')  % Frequency sync
run('exercise_4_2_2.m')  % Text decoding
run('exercise_4_2_3.m')  % Frame decoding

% Concatenated coding systems
run('exercise_4_3_1.m')  % Convolutional frames
run('exercise_4_3_2.m')  % RS frames  
run('exercise_4_3_3.m')  % Combined coding
```

## Key Results

**Channel Coding Performance:**
- Hamming (15,11): ~4dB coding gain at 10^-4 BER
- CRC-16: >99% error detection at moderate SNR
- Convolutional: ~6dB coding gain with Viterbi decoding
- Reed-Solomon: Powerful burst error correction

**Frequency Synchronization:**
- Maximum offset: ±3906 Hz for given parameters
- Least squares estimation using repeated preambles
- Works with QPSK training sequences

**Over-the-Air Processing:**
- Successfully processes captured signals on frequencies A-D
- Handles frame synchronization and data extraction
- Demonstrates practical SDR implementation

## Technical Implementation

**Hamming Codes:**
- Systematic (15,11) implementation
- Single error correction capability
- Syndrome-based decoding

**CRC-16:**
- DAB standard polynomial: x^16 + x^12 + x^5 + 1
- Error detection only (no correction)
- High detection rate for random errors

**Convolutional Codes:**
- Rate 1/3, constraint length 3
- Viterbi maximum likelihood decoding
- Trellis-based state machine implementation

**Reed-Solomon:**
- (255,239) code over GF(2^8)
- Systematic encoding with 16 parity symbols
- Corrects up to 8 symbol errors

**Concatenated Codes:**
- Inner convolutional + outer Reed-Solomon
- Superior performance for burst and random errors
- Used in DAB and other digital broadcasting systems

This practical demonstrates the trade-offs between different coding schemes and their practical implementation in software-defined radio systems.