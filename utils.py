"""
CSC 471 Final Project
November 10, 2025

This project is a toy simulation of Shors algorithm. Shor's algorithm is a quantum algorithm used for faster prime
factorization, but is limited by real world constraints such as error-prone qubits and a reliance on a high number
of qubits which is not currently feasible at this time. My project uses Qiskit’s quantum simulator to demonstrate the
core ideas of the algorithm and how it works to decrypt RSA on small non-realistic RSA numbers like (n=15 or n=95)

Sources:
https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/shor.ipynb
https://www.geeksforgeeks.org/dsa/shors-factorization-algorithm - I didn't end up using an actual quantum computer but saw the potential implementation and helpful to read
https://medium.com/@aa.adnane/breaking-rsa-understanding-the-algorithm-and-exploring-vulnerabilities-and-the-quantum-threat-cbd148ee7f3a

Necessary Installations are:
qiskit, qiskit_aer, and
"""


import random
import numpy as np
from math import gcd
from typing import Optional, Tuple
from fractions import Fraction
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator




def gcd_classical(a: int, b: int) -> int:
    return gcd(a, b) #Calculates the GCD in the standard way

#Uses the r (period of the function a^x % N), a smaller coprime to N, and N to calculate non-trivial factors
def find_factors_from_period(r: Optional[int], a: int, N: int) -> Tuple[Optional[int], Optional[int]]:
    if r is None or r <= 0 or r % 2 != 0: #Ensures r is valid - for Shors to work
        return None, None

    x = pow(a, r // 2, N) #Computes x that is based on the key property in Shors algo.

    if x == N - 1:
        return None, None

    # Calculate the two factor guesses
    factor1 = gcd_classical(x - 1, N)
    factor2 = gcd_classical(x + 1, N)

    # Validate and return the non-trivial factors
    p, q = None, None
    if factor1 > 1 and factor1 < N:
        p = factor1
    if factor2 > 1 and factor2 < N:
        q = factor2

    if p and q:
        return p, q
    elif p:
        return p, N // p
    elif q:
        return q, N // q

    return None, None

#Quantum Part:

#Computes Inverse Quantum Fourier Transform - used to find the period (r)
def qft_dagger(n):
    qc = QuantumCircuit(n) #Creates a quantum circuit with n qubits and 0 classical bits.
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1) #Swaps qubits so order is reversed
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / float(2 ** (j - m)), m, j) #This applies a controlled phase gate only to ket one - excited qubit, and it slightly twists the target qubit to help reveal the hidden pattern.
        qc.h(j) #Turns the phase information of the qubits since they can be between ket one and ket 0 (superposition) into a pattern which is the period.
    qc.name = "QFT†"
    return qc


def c_amod15(a, power):
    """
   The number 15 is hardcoded because this is a simplified demonstration of Shor’s algorithm for small numbers.
   Hardcoding allows the modular multiplication to be implemented with simple qubit swaps and Pauli-X gates rather
   than a full general modular exponentiation circuit, which would be more complex and require more qubits.
    """
    if a not in [2, 4, 7, 8, 11, 13]:
        raise ValueError(f"Base 'a'={a} is not supported by the hardcoded c_amod15 circuit for N=15.")

    #Creates a 4 qubit quantum circuit
    U = QuantumCircuit(4)
    for _iteration in range(power):
        #Rearrange the 4 qubits to represent how the number changes when multiplied by a mod 15
        if a in [2, 13]:  #Shifts all bits to the right performs multiplication by 2 (mod 15).
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        if a in [7, 8]: # Shifts all the bits to the left - performs multiplication by 7 or 8 (mod 15).
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        if a in [4, 11]: #Cross connect bits gives the same effect as multiplying by 4 or 11 mod 15.
            U.swap(1, 3)
            U.swap(0, 2)
        if a in [7, 11, 13]:
            for q in range(4):
                U.x(q) #After the swaps, it applies Pauli-X gates to the qubit - meaning it flips all the qubit stats so ket 1 <--> ket 0

   #Turns into a gate which can be used in a larger circuit
    U = U.to_gate()
    U.name = f"U({a}^{power} mod 15)"

    #Makes the gate controlled so it can only run at ket 1.
    c_U = U.control()
    return c_U


def shors_period_finding_qiskit(N: int, a: int, N_COUNT: int = 8) -> Optional[int]:
    """
     Runs the Quantum Phase Estimation (QPE) simulation to find the period r for a^x mod N.
    The period r is the key to factoring N in Shor’s algorithm.
    """
    try:
        # Creates a new circuit with N_COUNT for the phase estimation part and 4 for the number of qubits being multiplied
        qc = QuantumCircuit(N_COUNT + 4, N_COUNT)
        qc.h(range(N_COUNT)) #Uses hadamard gates - all possible options for a explored in parallel.
        qc.x(N_COUNT) #Initializes lower register as ket 1 meaning - starts modular multiplication from 1.

        #Applies controlled modular multiplication gates
        for q in range(N_COUNT):
            exponent = 2 ** q
            qc.append(c_amod15(a, exponent), #Adds the controlled modular multiplication gate to the circuit - does a^exponent (mod 15)
                      [q] + [i + N_COUNT for i in range(4)]) #Uses the 4 additional qubits.

        qc.append(qft_dagger(N_COUNT), range(N_COUNT)) #Applies inverse QFT to find the pattern to determine the period (r)
        qc.measure(range(N_COUNT), range(N_COUNT)) # Measure the phase qubits – converts wave states into 0s and 1s that reveal the hidden pattern (period r)

        # Simulation - Simulates a quantum computer
        aer_sim = AerSimulator()
        job = aer_sim.run(transpile(qc, aer_sim), shots=1, memory=True) #Runs the simulation
        readings = job.result().get_memory()


        measured_output = readings[0] #Takes only result from the list
        decimal = int(measured_output, 2) #Converts binary into base-10
        phase = decimal / (2 ** N_COUNT)  #Converts into fractional phase between 0 and 1 since superposition

        frac = Fraction(phase).limit_denominator(N) #Finds a rational fraction that approximates the quantum-measured phase
        r_guess = frac.denominator #Extracts the period, the denominator

        return r_guess #Returns period

    except ValueError as e:
        # Handles cases where 'a' is not supported by the hardcoded c_amod15
        print(f"  [Error] Skipping base a={a}. Reason: {e}")
        return None
    except Exception as e:
        print(f"  [Error] Simulation failed for base a={a}. {e}")
        return None


# --- 3. Full RSA Cracking Driver ---

def crack_rsa(N: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Repeatedly runs Shors algo until a non-trivial factor is found.
    """

    #Checks N's value
    if N < 4:
        print("N must be a composite number greater than 3.")
        return None, None

    max_attempts = 100
    attempt = 0

    while attempt < max_attempts:
        attempt += 1

        # Select a random base 'a' such that 1 < a < N
        a = random.randint(2, N - 1)

        print(f"Attempt {attempt}: Testing base a = {a}...")

        # Check if gcd(a, N) > 1
        g = gcd_classical(a, N)
        if g > 1 and g < N:
            print(f"  [Classical Success] Found factor {g} (via GCD check) on attempt {attempt}.")
            return g, N // g

        # Run QPE simulation to find period r
        r = shors_period_finding_qiskit(N, a)

        if r is None:
            continue

        print(f"  [Quantum Result] Measured period r = {r}")

        # 4. Classical Post-Processing: Use r to find factors
        factor1, factor2 = find_factors_from_period(r, a, N)

        if factor1 and factor2:
            print(f"  [Success] Non-trivial factors found: {factor1} and {factor2}")
            return factor1, factor2

        print("  [Failure] Period r did not yield factors. Retrying...")

    print(f"\n[FAILURE] Exceeded {max_attempts} attempts without finding a factor.")
    return None, None

