'''
CSC 471 Final Project
November 10, 2025

This file acts as the UI for the simulation and also has a tab that explain Shor's algorithm.

Sources:
https://docs.streamlit.io/develop/quick-reference/cheat-sheet - Streamlit documentation overall but this cheat sheet helped.
'''

import streamlit as st
from utils import crack_rsa

st.set_page_config(page_title="RSA Cracking Demo", layout="centered")
st.title("RSA Factoring Demo")

# Create tabs: Factoring and Shor Explanation
tab1, tab2 = st.tabs(["Simulation", "Shor's Algorithm Explained"])

# Simulation
with tab1:
    st.header("Factor an RSA modulus N")

    N = st.number_input(
        "Enter RSA modulus N (composite integer > 3). N Must be The Product of 2 Distinct Prime factors:",
        min_value=4, value=15, step=1
    )

    log_messages = []
    def st_log(msg):
        log_messages.append(msg)

    if st.button("Find Factors"):
        st.write(f"Running factorization on N = {N} ...")
        log_messages.clear()

        P, Q = crack_rsa(N)

        if log_messages:
            st.subheader("Logs")
            for msg in log_messages:
                st.write(msg)

        if P and Q:
            P, Q = min(P, Q), max(P, Q)
            phi_N = (P - 1) * (Q - 1)

            st.success(f"Factors found: P = {P}, Q = {Q}")
            st.info(f"Totient φ(N) = {phi_N}")
            st.write("This value can be used to derive the private decryption key.")
        else:
            st.error("Failed to factor N with the current method.")

# Tab 2: Explanation
with tab2:
    st.header("Understanding Shor's Algorithm")
    st.subheader("What is Shor's Algorithm?")
    st.markdown("""
    Shor's algorithm is a quantum algorithm for quickly finding the prime factors of a large algorithm.
    This is important because RSA (a cryptographic algorithm) is built on the principle that classical
    computing does not perform prime factorization quickly. This is since classical computers take
    exponential time, whereas Shor's algorithm can find factors in polynomial time. And since Shor's
    algorithm is significantly faster, it will revolutionize cryptography.
    """)

    st.subheader("High-Level Steps")
    st.markdown("""
    1. **Number to be factorized (N)** Can't be a prime, not an even number, and cannot be a base raised to a power.
    2. **Pick a random number `a`** less than `N`.
    3. **Check GCD**: If `gcd(a, N) > 1`, you already found a factor.
    4. **Quantum Phase Estimation**:
       - Find the **period `r`** of the function `f(x) = a^x mod N`.
       - This is done using a quantum circuit with the Quantum Fourier Transform.
    5. **Classical Post-Processing**:
       - If `r` is even and `a^(r/2) ≠ -1 mod N`, then
         ```
         factor1 = gcd(a^(r/2) - 1, N)
         factor2 = gcd(a^(r/2) + 1, N)
         ```
       - These give non-trivial factors of N.
    6. Repeat until factors are found.
    """)

    st.subheader("Key Quantum Concepts")
    st.markdown("""
    - **Superposition**: Quantum bits (qubits) can represent multiple values at once.
    - **Entanglement**: Qubits can be correlated, enabling interference patterns.
    - **Quantum Fourier Transform (QFT)**: Extracts the period of a function efficiently.
    - **Period Finding**: The heart of Shor's algorithm; classical post-processing then yields factors.
    """)

    st.subheader("References / Further Reading")
    st.markdown("""
    - https://www.geeksforgeeks.org/dsa/shors-factorization-algorithm/
    - https://kaustubhrakhade.medium.com/shors-factoring-algorithm-94a0796a13b1
    - https://www.classiq.io/insights/shors-algorithm-explained
    - https://uwaterloo.ca/institute-for-quantum-computing/outreach/quantum-101/glossary
    - https://courses.physics.illinois.edu/phys498cmp/sp2022/QC/QFT.html
    - https://milvus.io/ai-quick-reference/what-are-the-basic-quantum-gates-hadamard-pauli-etc
    """)
