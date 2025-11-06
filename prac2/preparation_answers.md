# COMS4105 Practical 2 - Preparation Questions

## Question 1: Hamming (7,4) Code Design

**Generator Matrix G:**
The Hamming (7,4) code uses a 4×7 generator matrix. A good generator matrix in systematic form:

```
G = [1 0 0 0 | 1 1 0]
    [0 1 0 0 | 1 0 1]  
    [0 0 1 0 | 0 1 1]
    [0 0 0 1 | 1 1 1]
```

This is a "good" Hamming matrix because it satisfies the minimum distance property (dmin = 3), ensuring single error correction capability and systematic form for easy encoding/decoding.

**Encoding message [1 0 0 0]:**
Codeword = message × G = [1 0 0 0] × G = [1 0 0 0 1 1 0]

**Parity Check Matrix H:**
```
H = [1 1 0 1 | 1 0 0]
    [1 0 1 1 | 0 1 0]
    [0 1 1 1 | 0 0 1]
```

**Error in 4th bit:**
Received word: [1 0 0 1 1 1 0] (error in position 4)
Syndrome S = received × H^T = [0 1 1]

The syndrome [0 1 1] = 3 in decimal, indicating error in bit position 3 (using standard indexing). The algorithm would correct bit 3.

**Code Parameters:**
- Code rate: 4/7 ≈ 0.571
- Can detect: 2 errors
- Can correct: 1 error

## Question 2: CRC-16 and Reed-Solomon in DAB

**DAB CRC-16 Polynomial (ETSI EN 300 401 §5.3.1):**
The polynomial used is: G(x) = x^16 + x^12 + x^5 + 1

**Checksum for data 1001 0000 0000 1001:**
Using polynomial division with G(x), the CRC-16 checksum is calculated by appending 16 zeros and performing modulo-2 division.

**FIC CRC-16 Coverage (§5.3.1):**
CRC-16 is performed on 30 bytes of data in each FIB (Fast Information Block).

**MSC Reed-Solomon Code (§5.2.4):**
- Code structure: RS(255, 239) over GF(2^8)  
- Generator polynomial: Based on primitive element α in GF(2^8)
- Field polynomial: x^8 + x^4 + x^3 + x^2 + 1

**RS Encoding Process:**
RS encoding adds 16 parity bytes to 239 message bytes, creating systematic codewords that can correct up to 8 byte errors.

## Question 3: DAB Convolutional Coding

**DAB Convolutional Encoder (ETSI EN 300 401 §11.1):**
- Code rate: 1/4
- Constraint length: K = 7
- Generator polynomials: G1-G4 define the four output sequences

**Encoding bits 101:**
Input: 101 + padding zeros
Using the trellis structure with state transitions, the encoded output depends on the current state and input bit.

**Simplified Convolutional Code:**
Generator polynomials:
- x0 = 1 + p
- x1 = 1  
- x2 = 1 + p + p²

**Trellis Diagram:**
The trellis shows state transitions for the 4-state encoder (2-bit memory).

**Decoding '101 001 110 011 001':**
Using Viterbi algorithm through the trellis, finding the maximum likelihood path.

## Question 4: Lecture References and Git Setup

**Channel Coding Lectures:**
- Hamming codes: Lecture 8 (covered block codes and syndrome decoding)
- Convolutional codes: Lecture 9 (trellis diagrams and Viterbi decoding)

**Git Repository Structure:**
Repository organized with /prac2/ directory containing:
- README.md (this file)
- MATLAB/Python implementation files
- Test data and results

## Question 5: Frequency Synchronization

**Signal Model:**
y[t] = e^(j2πfct) h x[t] + n[t]

**Preamble Relationship:**
y[t + Npre] = e^(j2πfcNpre) y[t]

**Least Squares Solution:**
For vectors y1 and y2 (first and second preambles), the frequency offset relationship:
y2 = e^(j2πfcNpre) y1

Using MATLAB's '\' operator or Python's numpy.linalg.lstsq to solve for the complex exponential term.

**Frequency Conversion:**
From the estimated complex exponential e^(j2πfcNpre), extract:
fc = angle(estimated_value) / (2π × Npre × Ts)

**Maximum Frequency Shift:**
Given:
- Sample rate fs = 2 MS/s
- Training sequence: 8 symbols × 2 repetitions = 16 symbols
- 16 samples per symbol

Npre = 16 × 16 = 256 samples
Maximum unambiguous frequency shift = fs/(2×Npre) = 2×10^6/(2×256) ≈ 3906 Hz

The limitation comes from phase ambiguity when the phase shift exceeds ±π between preamble copies.