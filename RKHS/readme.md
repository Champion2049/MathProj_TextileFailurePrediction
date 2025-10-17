# 🧮 Reproducing Kernel Hilbert Space (RKHS) and Reproducing Kernel (RK)

## 📘 Overview

A **Reproducing Kernel Hilbert Space (RKHS)** is a Hilbert space of functions where evaluation at each point can be represented as an inner product.  
Its defining feature is the **reproducing property**, which connects function evaluation to inner products through a **kernel function**.

---

## 🌱 1. What is an RKHS?

An RKHS is a Hilbert space 𝓗 of functions f(x) such that for every x, the *evaluation functional*  
Lₓ(f) = f(x)  
is **bounded (continuous)**.

By the Riesz Representation Theorem, this implies that for every x, there exists a unique function kₓ ∈ 𝓗 satisfying:

> f(x) = ⟨f, kₓ⟩ₕ  for all f ∈ 𝓗

This is the **reproducing property**.

---

## 🎯 2. The Reproducing Kernel (RK)

The **reproducing kernel** is a function:

> k(x, y) = ⟨kᵧ, kₓ⟩ₕ

It must satisfy:

1. **Symmetry:** k(x, y) = k(y, x)  
2. **Positive Semi-definiteness (PSD):** ΣᵢΣⱼ cᵢcⱼ k(xᵢ, xⱼ) ≥ 0  for all real cᵢ.

---

## 🧭 3. Constructing the RKHS from a Kernel

Given a PSD kernel k(x, y):

1. Start with finite linear combinations:  
   f(x) = Σᵢ αᵢ k(xᵢ, x)

2. Define the inner product:  
   ⟨f, g⟩ₕ = ΣᵢΣⱼ αᵢβⱼ k(xᵢ, xⱼ)

3. Complete the space under this inner product norm.

Each f in 𝓗 is thus a linear (possibly infinite) combination of kernels.

---

## ⚙️ 4. RKHS in Machine Learning

Kernel-based algorithms (e.g., **SVMs**, **Gaussian Processes**, **Kernel Ridge Regression**) operate implicitly in an RKHS.  
They avoid explicit mapping into high-dimensional space by using the **kernel trick**:

> ⟨φ(x), φ(y)⟩ₕ = k(x, y)

This allows computing inner products in 𝓗 without knowing the explicit mapping φ.

---

## 🌌 5. Common Examples of Kernels and Their RKHS

| **Kernel Function** | **Formula** | **RKHS Description** |
|----------------------|-------------|----------------------|
| **Linear** | k(x, y) = xᵀy | Space of linear functions f(x) = wᵀx |
| **Polynomial** | k(x, y) = (xᵀy + c)ᵈ | Polynomials up to degree d |
| **RBF (Gaussian)** | k(x, y) = exp(−‖x − y‖² / (2σ²)) | Infinitely smooth functions |
| **Laplacian** | k(x, y) = exp(−‖x − y‖ / σ) | Functions with bounded variation |
| **Sigmoid** | k(x, y) = tanh(a xᵀy + b) | Not always PSD (depends on a, b) |

---

## 🧩 6. Relationship Between RKHS, RK, and RBF

- **RKHS:** The complete function space defined by the kernel’s inner product.  
- **RK:** The kernel function k(x, y) that defines similarity.  
- **RBF Kernel:** A specific reproducing kernel based on Gaussian distance.  
- The RBF kernel’s RKHS is infinite-dimensional and contains smooth, continuous functions.

---

## 🧠 7. Concept Summary

| Concept | Meaning | Analogy |
|----------|----------|----------|
| **RKHS** | Space of functions with a kernel-defined inner product | “Feature space” of smooth functions |
| **Reproducing Kernel** | Function defining similarity between data points | “Dot product” in the RKHS |
| **RBF Kernel** | One example of a reproducing kernel | Gaussian similarity in infinite space |
| **Kernel Trick** | Compute inner products without explicit feature mapping | Shortcut for inner products |

---

## 🧭 8. Visual Intuition (Conceptual)

