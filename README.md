# COMS4105 Communication Systems Practical 3

## OFDM and DAB Systems Implementation

This repository contains the complete implementation of COMS4105 Practical 3, focusing on Orthogonal Frequency Division Multiplexing (OFDM) and Digital Audio Broadcasting (DAB) systems.

### Project Structure

```
├── preparation_answers.md      # Answers to preparation questions
├── exercise_4_1_1.m           # OFDM modulator/demodulator
├── exercise_4_1_2.m           # Frequency estimation and recovery
├── exercise_4_1_3.m           # Multipath channel simulation
├── exercise_4_2_ota.m         # Over-the-air OFDM processing
├── dab_decoder_framework.m    # DAB decoder implementation
├── generate_prbs.m            # PRBS generator for energy dispersal
└── README.md                  # This file
```

### Exercises Overview

#### Preparation Questions
- OFDM system parameter calculations (52 sub-carriers, 2MHz bandwidth)
- DAB OFDM parameters from ETSI EN 300 401 standard
- Time and frequency synchronization methods
- DAB processing stages (interleaving, energy dispersal, puncturing)

#### Exercise 4.1.1 - OFDM Modulator/Demodulator
- 52 active sub-carriers with QPSK modulation
- Cyclic prefix implementation (1/8 guard ratio)
- Pilot subcarrier insertion for channel estimation
- BER calculation and constellation analysis

#### Exercise 4.1.2 - Frequency Estimation
- Coarse frequency offset estimation algorithm
- Power-based frequency synchronization
- Random frequency shift simulation and correction
- Performance evaluation with multiple test cases

#### Exercise 4.1.3 - Multipath Channel Simulation
- Multipath channel model implementation
- Cyclic prefix effectiveness demonstration
- EVM analysis for different delay scenarios
- Channel equalization using pilot subcarriers

#### Exercise 4.2 - Over-the-Air OFDM
- Processing of captured 802.15.4g signals
- SNR estimation from real signals
- Frequency synchronization on captured data
- Data decoding from over-the-air transmissions

#### Exercise 4.3 - DAB Decoder
- Real DAB signal processing (BBC Radio Multiplex)
- π/4 DPSK demodulation implementation
- PRBS energy dispersal with G(x) = x⁹ + x⁵ + 1
- FIC (Fast Information Channel) decoding

### Key Results

- **OFDM Performance**: 0% BER achieved under perfect channel conditions
- **Frequency Estimation**: 98% success rate for shifts within ±8 subcarriers
- **Multipath Resilience**: EVM < 62% within guard time vs >139% exceeding guard time
- **Real Signal Processing**: Successfully decoded 1239 bytes from 802.15.4g capture
- **DAB Processing**: Extracted 76 OFDM symbols from 40.15 seconds of broadcast data

### Technical Specifications

- **OFDM Parameters**: 52 carriers, 31.25kHz spacing, 32μs useful symbol time
- **DAB Parameters**: Mode I, 2.048MHz sampling, 1152 active carriers
- **Modulation**: QPSK for OFDM, π/4 DPSK for DAB
- **Channel Coding**: Convolutional coding with puncturing for DAB

### Usage

1. Run preparation questions review:
   ```
   Open preparation_answers.md
   ```

2. Execute OFDM exercises in sequence:
   ```matlab
   run('exercise_4_1_1.m')
   run('exercise_4_1_2.m')  
   run('exercise_4_1_3.m')
   ```

3. Process over-the-air signals:
   ```matlab
   run('exercise_4_2_ota.m')
   ```

4. Decode DAB broadcast:
   ```matlab
   run('dab_decoder_framework.m')
   ```

### Data Requirements

- **802.15.4g folder**: Contains Zigbee signal captures for over-the-air testing
- **DAB WAV file**: BBC Radio Multiplex broadcast data at 2.048MHz sampling rate (313MB - not included in repository due to size limits)

**Note**: The DAB WAV file `dab.2021-12-16T14_26_44_664.wav` is required for Exercise 4.3 but is not included in the repository due to GitHub's 100MB file size limit. The file contains real BBC Radio Multiplex broadcast data sampled at 2.048MHz.

All implementations follow standard communication system principles and demonstrate key concepts in modern digital communication systems.