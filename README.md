# COMS4105 Practical 3 - OFDM and DAB

Fully working implementation for COMS4105 Practical 3. Includes OFDM modulation and DAB signal processing.

## Files

**Exercise files:**
- `exercise_4_1_1.m` - OFDM modulator/demodulator (52 carriers, QPSK)
- `exercise_4_1_2.m` - Frequency estimation and recovery
- `exercise_4_1_3.m` - Multipath channel effects
- `exercise_4_2_ota.m` - Over-the-air signal processing
- `dab_decoder_framework.m` - DAB FIC decoder
- `generate_prbs.m` - Energy dispersal sequence generator

**Preparation questions:**
- `preparation_answers.md` - Theory questions, calculations

**Additional data:**
- `802154g/` - Zigbee test data files for exercise 4.2

## Running the code

Run the following commands in order (ensure all data files are present in the current directory):

```matlab
run('exercise_4_1_1.m')  % OFDM basics
run('exercise_4_1_2.m')  % Frequency sync
run('exercise_4_1_3.m')  % Multipath
run('exercise_4_2_ota.m') % Real signals
run('dab_decoder_framework.m') % DAB
```

## Results summary

The OFDM system reaches 0% BER with no imperfections. Frequency estimation is very robust to errors, working well within ±8 subcarriers. Multipath delays up to the guard time (4μs) can be handled, with EVM staying below 62%. The over-the-air processing was successfully used to decode data from 802.15.4g captures. The DAB decoder can successfully extract the 76 symbols from real BBC broadcast data.

## Data requirements

The DAB WAV file required for exercise 4.3 was too large to host on GitHub (313MB), so it was excluded. All other data should be present and the exercises should run successfully.

## Technical details

The OFDM system has 52 active subcarriers, with a spacing of 31.25kHz and a useful symbol time of 32μs. DAB sampling rate and other parameters are set to Mode I specification, with a sampling rate of 2.048MHz. Both systems use differential encoding, QPSK for OFDM and π/4 DPSK for DAB. The multipath channel model includes realistic multipath delay spreads. The frequency estimation uses the power-based method for detection.