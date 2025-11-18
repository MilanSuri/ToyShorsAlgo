"""
Milan Suri
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
qiskit, qiskit_aer, and numpy
"""

import random  # Used to select random numbers for testing potential factors
import numpy as np  # Provides mathematical functions like pi
from math import gcd  # Standard Python function to compute greatest common divisor
from typing import Optional, Tuple  # Type hints for better readability
from fractions import Fraction  # Helps convert decimal numbers into fractions
from qiskit import QuantumCircuit, transpile  # Quantum circuit creation and optimization
from qiskit_aer import AerSimulator  # Simulates a quantum computer on a classical computer


# --- Classical Part ---

def gcd_classical(a: int, b: int) -> int:
    """
    Computes the greatest common divisor (GCD) of two integers a and b.
    This is the classical method used to find trivial factors quickly.
    """
    return gcd(a, b)  #Finds GCD of a and b


def find_factors_from_period(r: Optional[int], a: int, N: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Given the period r (from the quantum part), compute potential factors of N.
    This is the classical post-processing step in Shor's algorithm.

    Steps:
    1. Ensure r is valid (non-zero, positive, even).
    2. Compute x = a^(r/2) mod N.
    3. Use gcd(x-1, N) and gcd(x+1, N) to get non-trivial factors.
    """
    if r is None or r <= 0 or r % 2 != 0:
        # Invalid period: cannot use this r to find factors
        return None, None

    x = pow(a, r // 2, N)  # Compute (a^(r/2)) mod N
    if x == N - 1:
        # If x = N-1, the factors are trivial (1 and N), so we skip
        return None, None

    # Compute non-trivial factors
    factor1 = gcd_classical(x - 1, N)
    factor2 = gcd_classical(x + 1, N)

    # Validate factors: they must be greater than 1 and less than N
    p, q = None, None
    if factor1 > 1 and factor1 < N:
        p = factor1
    if factor2 > 1 and factor2 < N:
        q = factor2

    # Return a tuple of factors, if found
    if p and q:
        return p, q
    elif p:
        return p, N // p
    elif q:
        return q, N // q
    return None, None  # No non-trivial factors found


# --- Quantum Part ---

def qft_dagger(n):
    """
    Creates the inverse Quantum Fourier Transform (QFT†) circuit.
    QFT† is used to extract the hidden period r in the quantum phase estimation step.

    - QFT† transforms the quantum phases into measurable probabilities which help find a pattern.
    """
    qc = QuantumCircuit(n)  # Create n-qubit quantum circuit

    # Reverse the order of qubits
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)  # Swapping qubits to correct order after QFT

    # Apply phase rotations and Hadamard gates
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / float(2 ** (j - m)), m, j)  # Change qubit phases based on other qubits
        qc.h(j)  # Hadamard gate begin superposition - making it be equally between |0> and |1>

    qc.name = "QFT†"
    return qc


def c_amod15(a, power):
    """
    Hardcoded modular multiplication circuit for N=15.
    In real Shor's algorithm, this would be a full modular exponentiation circuit.

    This simplification is only for demonstration:
    - Swaps and X gates mimic modular multiplication by a^power mod 15.
    - Only supports specific bases a because it's hardcoded.
    """
    if a not in [2, 4, 7, 8, 11, 13]:
        raise ValueError(f"Base 'a'={a} is not supported for N=15 demo.")

    U = QuantumCircuit(4)  # 4 qubits to represent numbers mod 15

    for _iteration in range(power):
        # Bit manipulations to mimic modular multiplication
        if a in [2, 13]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        if a in [7, 8]:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        if a in [4, 11]:
            U.swap(1, 3)
            U.swap(0, 2)
        if a in [7, 11, 13]:
            for q in range(4):
                U.x(q)  # Flip qubits to simulate multiplication

    # Convert to a gate and make it controlled (only executes if control qubit is 1)
    U = U.to_gate()
    U.name = f"U({a}^{power} mod 15)"
    c_U = U.control()
    return c_U


def shors_period_finding_qiskit(N: int, a: int, N_COUNT: int = 8) -> Optional[int]:
    """
    Quantum Phase Estimation (QPE) to find the period r of a^x mod N.
    Period r is key to factoring N using Shor's algorithm.

    Steps:
    1. Prepare counting qubits in superposition (all possibilities explored).
    2. Apply controlled modular multiplication based on base a.
    3. Apply inverse QFT to extract the period.
    4. Measure qubits to get classical value representing r.
    """
    try:
        # Counting register (N_COUNT) + number register (4 qubits)
        qc = QuantumCircuit(N_COUNT + 4, N_COUNT)

        qc.h(range(N_COUNT))  # Superposition for parallel exploration of values of r
        qc.x(N_COUNT)  # Initialize number register to 1 (starting value)

        # Apply controlled modular multiplication gates
        for q in range(N_COUNT):
            exponent = 2 ** q
            qc.append(c_amod15(a, exponent), [q] + [i + N_COUNT for i in range(4)])

        # Apply inverse QFT to counting qubits to reveal the period
        qc.append(qft_dagger(N_COUNT), range(N_COUNT))

        qc.measure(range(N_COUNT), range(N_COUNT))  # Measure to convert qubits into classical bits

        # Simulate the quantum circuit using Aer
        aer_sim = AerSimulator()
        job = aer_sim.run(transpile(qc, aer_sim), shots=1, memory=True)
        readings = job.result().get_memory()
        measured_output = readings[0]

        # Convert binary result to decimal, then to fractional phase
        decimal = int(measured_output, 2)
        phase = decimal / (2 ** N_COUNT)
        frac = Fraction(phase).limit_denominator(N)
        r_guess = frac.denominator  # Denominator gives the period r

        return r_guess

    except ValueError as e:
        print(f"  [Error] Skipping base a={a}. Reason: {e}")
        return None
    except Exception as e:
        print(f"  [Error] Simulation failed for base a={a}. {e}")
        return None


# --- Full RSA Cracking ---

def crack_rsa(N: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Attempts to factor a composite number N using Shor's algorithm.
    Combines classical and quantum steps:
    1. Try GCD classically first (cheap check for trivial factors).
    2. Use quantum simulation to find period r.
    3. Convert r to non-trivial factors.
    """
    if N < 4:
        print("N must be a composite number greater than 3.")
        return None, None

    max_attempts = 100
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        a = random.randint(2, N - 1)  # Random base selection
        print(f"Attempt {attempt}: Testing base a = {a}...")

        # Classical quick check
        g = gcd_classical(a, N)
        if g > 1 and g < N:
            print(f"  [Classical Success] Found factor {g} (via GCD check) on attempt {attempt}.")
            return g, N // g

        # Quantum period finding
        r = shors_period_finding_qiskit(N, a)
        if r is None:
            continue  # Skip to next attempt if quantum part failed
        print(f"  [Quantum Result] Measured period r = {r}")

        # Classical post-processing
        factor1, factor2 = find_factors_from_period(r, a, N)
        if factor1 and factor2:
            print(f"  [Success] Non-trivial factors found: {factor1} and {factor2}")
            return factor1, factor2

        print("  [Failure] Period r did not yield factors. Retrying...")

    print(f"\n[FAILURE] Exceeded {max_attempts} attempts without finding a factor.")
    return None, None
