# COMS4105 Practical 3 - OFDM and DAB

Complete implementation for COMS4105 Practical 3 covering OFDM modulation and DAB signal processing.

## Files

**Core exercises:**
- `exercise_4_1_1.m` - OFDM modulator/demodulator (52 carriers, QPSK)
- `exercise_4_1_2.m` - Frequency estimation and recovery 
- `exercise_4_1_3.m` - Multipath channel effects
- `exercise_4_2_ota.m` - Over-the-air signal processing
- `dab_decoder_framework.m` - DAB FIC decoder
- `generate_prbs.m` - Energy dispersal sequence generator

**Documentation:**
- `preparation_answers.md` - Theory questions and calculations
- `802154g/` - Zigbee test data for exercise 4.2

## Running the code

Execute in order:
```matlab
run('exercise_4_1_1.m')  % OFDM basics
run('exercise_4_1_2.m')  % Frequency sync
run('exercise_4_1_3.m')  % Multipath
run('exercise_4_2_ota.m') % Real signals
run('dab_decoder_framework.m') % DAB
```

## Results summary

The OFDM implementation achieves 0% BER under perfect conditions. Frequency estimation works reliably within ±8 subcarriers. Multipath delays up to the guard time (4μs) are handled well, with EVM staying below 62%. The over-the-air processing successfully decodes data from 802.15.4g captures. DAB decoder extracts 76 symbols from real BBC broadcast data.

## Data requirements

You need the DAB WAV file for exercise 4.3 but it's too big for GitHub (313MB). Everything else should work with the included data files.

## Technical details

OFDM uses 52 active subcarriers with 31.25kHz spacing and 32μs useful symbol time. DAB follows Mode I specification with 2.048MHz sampling rate. Both systems use differential encoding - QPSK for OFDM and π/4 DPSK for DAB. The multipath channel model includes realistic delay spreads and the frequency estimation uses power-based detection.