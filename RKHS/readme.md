# 🧮 Reproducing Kernel Hilbert Space (RKHS) and Reproducing Kernel (RK)

## 📘 Overview

A **Reproducing Kernel Hilbert Space (RKHS)** is a Hilbert space of functions where evaluation at each point can be represented as an inner product.  
Its defining feature is the **reproducing property**, which connects function evaluation to inner products through a **kernel function**.

---

## 🌱 1. What is an RKHS?

An RKHS is a Hilbert space \( \mathcal{H} \) of functions \( f : X \to \mathbb{R} \) such that for every \( x \in X \), the **evaluation functional**

\[
L_x(f) = f(x)
\]

is **bounded (continuous)**.  
By the **Riesz Representation Theorem**, this implies that for every \( x \), there exists a unique function \( k_x \in \mathcal{H} \) satisfying:

\[
f(x) = \langle f, k_x \rangle_{\mathcal{H}}, \quad \forall f \in \mathcal{H}.
\]

This is the **reproducing property**.

---

## 🎯 2. The Reproducing Kernel (RK)

The **reproducing kernel** is a function:

\[
k(x, y) = \langle k_y, k_x \rangle_{\mathcal{H}}.
\]

It must satisfy:

1. **Symmetry:** \( k(x, y) = k(y, x) \)
2. **Positive Semi-definiteness (PSD):**
   \[
   \sum_{i,j=1}^n c_i c_j k(x_i, x_j) \ge 0, \quad \forall c_i \in \mathbb{R}.
   \]

---

## 🧭 3. Constructing the RKHS from a Kernel

Given a PSD kernel \( k(x, y) \), the corresponding RKHS is constructed as follows:

1. Start with finite linear combinations:
   \[
   f(x) = \sum_{i=1}^n \alpha_i k(x_i, x)
   \]

2. Define the inner product:
   \[
   \left\langle \sum_i \alpha_i k(x_i, \cdot), \sum_j \beta_j k(x_j, \cdot) \right\rangle_{\mathcal{H}} = \sum_{i,j} \alpha_i \beta_j k(x_i, x_j)
   \]

3. Complete this space under the norm induced by the inner product.

Thus, each \( f \in \mathcal{H} \) is (possibly infinite) a linear combination of kernels.

---

## ⚙️ 4. RKHS in Machine Learning

Kernel-based algorithms (e.g., **SVMs**, **Gaussian Processes**, **Kernel Ridge Regression**) implicitly operate in an RKHS.  
They avoid explicitly mapping data into high-dimensional space by using the **kernel trick**:

\[
\langle \phi(x), \phi(y) \rangle_{\mathcal{H}} = k(x, y)
\]

This means we compute inner products in \( \mathcal{H} \) without knowing the explicit mapping \( \phi \).

---

## 🌌 5. Common Examples of Kernels and Their RKHS

| **Kernel Function** | **Formulation** | **RKHS Description** |
|----------------------|-----------------|----------------------|
| **Linear** | \( k(x, y) = x^\top y \) | Space of linear functions \( f(x) = w^\top x \) |
| **Polynomial** | \( k(x, y) = (x^\top y + c)^d \) | Polynomials up to degree \( d \) |
| **RBF (Gaussian)** | \( k(x, y) = \exp(-\|x - y\|^2 / 2\sigma^2) \) | Infinitely smooth functions |
| **Laplacian** | \( k(x, y) = \exp(-\|x - y\| / \sigma) \) | Functions with bounded variation |
| **Sigmoid** | \( k(x, y) = \tanh(a x^\top y + b) \) | Not always PSD (depends on parameters) |

---

## 🧩 6. Relationship Between RKHS, RK, and RBF

- **RKHS:** The entire function space equipped with an inner product defined by a kernel.  
- **RK:** The kernel function \( k(x, y) \) itself, defining how similarity is measured.  
- **RBF Kernel:** One specific type of reproducing kernel (Gaussian).  
- The RKHS of the RBF kernel is infinite-dimensional and contains very smooth functions.

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

