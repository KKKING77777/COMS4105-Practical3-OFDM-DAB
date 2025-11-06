# COMS4105 Practical 3 - Preparation Questions 

## Question 1: OFDM system 

**Provided parameters:**
- Number of sub-carriers = 52 
- Bandwidth = 2 MHz 
- Total symbol duration = 36 µs 
- Guard time = 1/8 of the useful symbol time
- QPSK modulation per sub-carrier 

**To calculate other parameters:**
- a. Subcarrier spacing = Useful symbol time⁻¹ 
- b. Maximum delay spread that can be tolerated = guard time

**Solution:** 

First, we need to know the relationship between the total symbol time and the useful symbol time. Since 36µs includes both guard time and FFT integration period, we assume the guard time is equal to 1/8 of the useful symbol time. We calculate the FFT integration period Tu is 32µs, the corresponding guard time is 4µs.

The subcarrier spacing can be determined by the useful symbol time. Δf = 1/Tu , we have the subcarrier spacing = 31.25kHz. In a multipath propagation environment, the cyclic prefix (guard time) determines the maximum delay spread that can be tolerated, thus the maximum delay spread = 4µs.

The total number of subcarriers in the system is 52, of which the actual number of used data subcarriers and pilot subcarriers is less than 52, since the guard bands may be used to protect against the out-of-band emissions, the used subcarriers are distributed in an actual bandwidth of less than 2MHz, it is 1.625MHz. Considering 4 subcarriers are pilots, the actual number of data subcarriers = 52 - 4 = 48. Each subcarrier is modulated with QPSK, providing 2 bits per subcarrier. The maximum data rate = 2 x 48 x 27777.78symbols/second = 2.67Mbps

## Question 2: DAB OFDM parameters (ETSI EN 300 401 §14.2)

ETSI EN 300 401 standard section 14.2, the DAB system in mode I uses the following OFDM parameters:

- The total number of subcarrier positions is 1536, 384 are null carriers, the number of active subcarriers is 1152
- FFT size K= 2048, guard interval length is 504 samples
- Time unit T = 1/2048000 seconds. This requires the sampling rate to be 2.048MHz. The useful symbol duration Tu is 1024 time units = 500µs. The guard interval duration Tg is 246 time units = 120µs. The total symbol duration Ts is 1270 time units = 620µs. The subcarrier spacing is 977Hz = 1kHz/1024 as per standard.

## Question 3: DAB synchronization methods 

The DAB system time synchronization process mainly relies on null symbol detection. In the transmission frame structure, the insertion of the null symbol generates a unique pattern in the frame structure, which is conducive to the subsequent detection. The transmission frame contains the Phase Reference Symbol PRS, Fast Information Channel FIC symbols, and Main Service Channel MSC symbols. The receiver can detect the period of the null symbol through energy detection or correlation method to look for known null symbol patterns. The structure of the Phase Reference Symbol is known, all subcarriers are modulated with a pseudo-random binary sequence. It is used as a differential demodulation reference and to maintain phase coherence for the subsequent symbols.

Frequency synchronization in the DAB system is mainly divided into coarse synchronization and fine synchronization.

Coarse frequency synchronization mainly includes three steps: First, it samples and receives signals in the full bandwidth of the DAB system at around the target frequency. Second, it performs FFT on the received signal and analyzes the FFT result. Finally, through coarse adjustments of frequency, it searches for the maximum energy concentration position on the expected subcarrier positions. According to the feature that the center carrier should be zero, it determines that the frequency correction range is converged. Fine frequency synchronization is performed by utilizing the periodic nature of the cyclic prefix, that is, the correlation between the guard interval and the tail of the useful symbol period. In the frequency domain, the correlation function reaches a peak at the exact frequency offset.

## Question 4: OFDM related lecture 

The theory related to OFDM is in lecture 16 and 17, include multi-carrier modulation principle and orthogonality condition, cyclic prefix and inter-symbol interference, frequency domain processing and so on.

Git repository is used to version control the experimental programs and results, the main content to initialize a new repository is to create a README.md file containing the description of experiments, configure .gitignore rules suitable for MATLAB files, create an organized directory structure for different exercises, etc.

## Question 5: DAB processing stages 

Frequency interleaving in the DAB system uses pseudo-random permutation tables to disperse the adjacent bits to different subcarriers, providing frequency diversity to combat selective fading. Interleaving table defines the mapping relationship between the logical subcarrier indices and the physical subcarrier positions. This randomization process effectively improves the transmission quality.

The energy dispersal mechanism uses a pseudo-random binary sequence PRBS for XOR operation with generator polynomial G(x) = x⁹ + x⁵ + 1. The energy dispersal operation is to randomize the data with the purpose of ensuring that the power spectral density is flat, it is typically applied before channel coding.

Convolutional coding uses rate 1/4 coding and constraint length 7 to provide error correction capability. The generator polynomials define the coding relationship and create redundancy to detect and correct errors. Puncturing is used to selectively remove certain coded bits to increase the code rate. Puncturing vectors in section 11 indicate the positions of transmitted bits or discarded bits, this flexible coding scheme allows for implementation of various punctured code rates such as 1/2, 2/3, 3/4, etc. The receiver processing includes first de-puncturing to insert neutral value, then Viterbi decoding for the whole error correction decoding.