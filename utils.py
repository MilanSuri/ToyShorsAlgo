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
qiskit, qiskit_aer, and numpy.
"""

import random
import numpy as np
from math import gcd
from typing import Optional, Tuple
from fractions import Fraction
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


#------------ Classical Functions ------------

def gcd_classical(a: int, b: int) -> int:
    """Return the greatest common divisor of a and b."""
    return gcd(a, b)


def find_factors_from_period(r: Optional[int], a: int, N: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Use period r to attempt recovery of non-trivial factors of N.
    """
    if r is None or r <= 0 or r % 2 != 0:
        return None, None

    x = pow(a, r // 2, N)
    if x == N - 1:
        return None, None

    factor1 = gcd_classical(x - 1, N)
    factor2 = gcd_classical(x + 1, N)

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


#------------ Quantum Components ------------

def qft_dagger(n):
    """
    Build inverse QFT on n qubits for phase estimation.
    """
    qc = QuantumCircuit(n)

    # Bit reversal
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)

    # Controlled phase rotations and Hadamard
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / float(2 ** (j - m)), m, j)
        qc.h(j)

    qc.name = "QFT†"
    return qc


def c_amod15(a, power):
    """
    Controlled modular multiplication a^power mod 15 (hard-coded for N=15).
    """
    if a not in [2, 4, 7, 8, 11, 13]:
        raise ValueError(f"Base 'a'={a} is not supported for N=15.")

    U = QuantumCircuit(4)

    # Apply permutation corresponding to multiplication by a^power mod 15
    for _iteration in range(power):
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
                U.x(q)

    U = U.to_gate()
    U.name = f"U({a}^{power} mod 15)"
    return U.control()


#------------ Period Finding ------------

def shors_period_finding_qiskit(N: int, a: int, N_COUNT: int = 8) -> Optional[int]:
    """
    Estimate the period r of a^x mod N using quantum phase estimation.
    """
    try:
        qc = QuantumCircuit(N_COUNT + 4, N_COUNT)

        # Counting register in superposition, work register |1>
        qc.h(range(N_COUNT))
        qc.x(N_COUNT)

        # Apply controlled-U operations
        for q in range(N_COUNT):
            exponent = 2 ** q
            qc.append(c_amod15(a, exponent),
                      [q] + [i + N_COUNT for i in range(4)])

        # Inverse QFT and measure
        qc.append(qft_dagger(N_COUNT), range(N_COUNT))
        qc.measure(range(N_COUNT), range(N_COUNT))

        aer_sim = AerSimulator()
        job = aer_sim.run(transpile(qc, aer_sim), shots=1, memory=True)
        readings = job.result().get_memory()

        measured_output = readings[0]
        decimal = int(measured_output, 2)
        phase = decimal / (2 ** N_COUNT)

        # Convert phase to rational; denominator is candidate period
        frac = Fraction(phase).limit_denominator(N)
        return frac.denominator

    except ValueError as e:
        print(f"  [Error] Skipping base a={a}. Reason: {e}")
        return None
    except Exception as e:
        print(f"  [Error] Simulation failed for base a={a}. {e}")
        return None


#------------ Full Shor ------------

def crack_rsa(N: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Attempt to factor N using Shor's algorithm simulation.
    """
    if N < 4:
        print("N must be a composite number > 3.")
        return None, None

    max_attempts = 100
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        a = random.randint(2, N - 1)
        print(f"Attempt {attempt}: Testing base a = {a}...")

        # Classical shortcut using gcd
        g = gcd_classical(a, N)
        if g > 1 and g < N:
            print(f"  [Classical Success] Found factor {g}.")
            return g, N // g

        # Quantum period estimation
        r = shors_period_finding_qiskit(N, a)
        if r is None:
            continue

        print(f"  [Quantum Result] Measured period r = {r}")

        # Convert period to candidate factors
        factor1, factor2 = find_factors_from_period(r, a, N)
        if factor1 and factor2:
            print(f"  [Success] Non-trivial factors: {factor1} and {factor2}")
            return factor1, factor2

        print("  [Failure] Period did not yield factors. Retrying...")

    print(f"\n[FAILURE] Exceeded {max_attempts} attempts without finding factors.")
    return None, None
