# COMS4105 Practical 3 - Preparation Questions

## Question 1: OFDM System Parameters

Given:
- 52 sub-carriers
- Bandwidth: 2 MHz  
- Total symbol duration: 36 µs
- Guard time: 1/8 of useful symbol time
- QPSK modulation per sub-carrier

In the given OFDM system configuration, we first need to understand the relationship between total symbol duration and useful symbol time. The total symbol duration of 36µs includes both guard time and FFT integration period, where the guard time equals 1/8 of the useful symbol time. Through calculation, the FFT integration period Tu is 32µs with corresponding guard time of 4µs.

The subcarrier spacing is directly determined by the useful symbol time. Using the formula Δf = 1/Tu, the subcarrier spacing is calculated as 31.25kHz. In multipath propagation environments, the cyclic prefix (guard time) determines the maximum delay spread the system can tolerate, therefore the maximum delay spread is 4µs.

The system uses 52 subcarriers in total, occupying an actual bandwidth of 1.625MHz, which is less than the allocated 2MHz bandwidth, leaving space for guard bands. Considering 4 subcarriers are used as pilots, there are 48 actual data subcarriers. Each subcarrier employs QPSK modulation providing 2 bits, combined with a symbol rate of 27777.78 symbols/second, the maximum data rate can be calculated as 2.67Mbps.

## Question 2: DAB OFDM Parameters (ETSI EN 300 401 §14.2)

According to ETSI EN 300 401 standard section 14.2, the DAB system in Mode I employs specific OFDM parameter configurations. The system defines 1536 total subcarrier positions, but 384 are null carriers, resulting in 1152 active subcarriers. The FFT size is set to 2048 with a guard interval of 504 samples.

Regarding timing parameters, DAB uses a special time unit T = 1/2048000 seconds, corresponding to a 2.048MHz sampling rate requirement. The useful symbol duration Tu is 1024 time units equaling 500µs, guard interval duration Tg is 246 time units equaling 120µs, and total symbol duration Ts is 1270 time units equaling 620µs. The subcarrier spacing is 977Hz, calculated as 1kHz divided by 1024 according to the standard.

## Question 3: DAB Synchronization Methods

DAB system time synchronization primarily relies on null symbol detection, which creates distinctive patterns in the transmission frame structure. The transmission frame contains Phase Reference Symbol PRS, Fast Information Channel FIC symbols, and Main Service Channel MSC symbols. Receivers can detect null symbol periods through energy detection methods that look for power drops, or use correlation methods with known null symbol patterns. The Phase Reference Symbol has a known structure where all subcarriers are modulated with a pseudo-random binary sequence, providing a differential demodulation reference and maintaining phase coherence for subsequent symbols.

Frequency synchronization consists of coarse and fine synchronization stages. Coarse frequency synchronization first captures full bandwidth signals around the expected frequency, performs FFT on the received signal, then looks for energy concentration in expected subcarrier positions. By adjusting frequency until maximum energy aligns with active subcarriers, using the characteristic that the center carrier should be zero for frequency correction. Fine frequency synchronization utilizes the periodic nature of the cyclic prefix, correlating the guard interval with the end of the useful symbol period, where correlation peaks indicate precise frequency offset.

## Question 4: OFDM Lecture Reference

OFDM-related content was covered in detail in lectures 16 and 17 of the course, including multi-carrier modulation principles, orthogonality conditions, cyclic prefix and inter-symbol interference, and frequency domain processing. These theoretical foundations are important for understanding OFDM modulation/demodulation, frequency synchronization, and multipath channel analysis in this experiment.

Regarding Git repository establishment, it requires creating a README.md file containing experimental descriptions, configuring .gitignore rules suitable for MATLAB files, and establishing organized directory structures for different exercises. Such version control setup helps with code management and tracking experimental results.

## Question 5: DAB Processing Stages

DAB system frequency interleaving uses pseudo-random permutation tables to spread adjacent bits across different subcarriers, providing frequency diversity against selective fading. The interleaving table establishes mapping relationships between logical subcarrier indices and physical subcarrier positions, and this randomization process effectively improves transmission quality.

The energy dispersal mechanism uses pseudo-random binary sequence PRBS for XOR operations with generator polynomial G(x) = x⁹ + x⁵ + 1. This process aims to randomize data ensuring flat power spectral density, typically applied before channel coding. Energy dispersal helps avoid adverse effects from periodic patterns that may exist in data on transmission performance.

Convolutional coding employs rate 1/4 coding with constraint length 7, providing error correction capability through generator polynomials and creating redundancy for error detection and correction. Puncturing techniques selectively remove coded bits to increase code rate, with puncturing vectors in section 11 specifying transmitted or discarded bit positions. This flexible coding scheme allows implementation of various code rates like 1/2, 2/3, 3/4, etc. The receiver processing involves first de-puncturing by inserting neutral values, then performing Viterbi decoding to complete the entire error correction decoding process.