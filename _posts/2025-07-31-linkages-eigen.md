---
title: "When 4-Bar Linkages Break Down: An Intuitive Linear Algebra Perspective"
date: 2025-07-29
categories: [blog]
tags: [linkages, robotics, kinematics, linear-algebra, manifold-optimization, JAX]
layout: blog
comments: true
excerpt: "We explore how linear algebra, specifically eigenvalue analysis of the constraint Hessian, reveals the locking behavior of planar 4-bar linkages."
---

# When 4-Bar Linkages Break Down: An Intuitive Linear Algebra Perspective

In this post, I share insights from building a custom 2D planar linkage solver and how I used it to analyze when and why a linkage becomes **locked** or **partially locked**. Understanding the **algebraic signature** of such behavior helps us design better mechanisms and even train generative models to avoid pathological configurations. More importantly, we gain an appreciation of the infinitely diverse design space of planar linkage mechanisms.

This blog provides an **algebraic lens** through which to view the motion (and failure) of 4-bar linkage mechanisms.

---

## Motivation

It is important to understand how to algebraically characterize a locked or partially locked linkage mechanism, so that in future work, we can potentially inject these algebraic properties in a loss or reward function to mitigate invalid linkage designs during linkage generation.

In this work, we unroll the specific solving of joint positions at every step in a full rotation of the crank link by walking in the optimization landscape in the direction of the eigenvector corresponding to the zero eigenvalue of the **Hessian of the constraint function**, using a tunable step size. In doing so, we gain a rich understanding of how the eigenvalue evolves depending on the initial linkage configuration.

---

## Locked Linkage Analysis in Conventional Linear Algebra

To understand locking behavior in planar linkage mechanisms, we inspect the **null space of the Hessian** of the constraint function.

The linkage configuration is a set of **fixed-length bars** constrained by **joint positions**. Our goal is to move the mechanism along a _minima river_ — the set of all valid configurations that satisfy the constraint. Deviating off that river collapses a degree of freedom and can lead to a locking event.

We define the **constraint function** as a differentiable objective measuring deviation from fixed link lengths:

$$
\text{loss} = \sum_{(i,j) \in E} \left( l_{ij} - \| \mathbf{x}_i - \mathbf{x}_j \| \right)^2
$$

where:

- $\mathbf{x}_i$, $\mathbf{x}_j$ $\in$ $\mathbb{R}^2$ are the 2D joint positions
- $l_{ij}$'s are the original link lengths

The loss is minimized when all link lengths are preserved during motion.

---

### Step-by-Step Eigenvalue-Based Locking Analysis

We iterate through the following steps from the initial linkage configuration:

1. **Compute the Hessian**  
   Compute the Hessian of the constraint function with respect to the **movable joints**. Fixed joints are excluded since they don’t contribute to curvature.

2. **Identify the zero eigenvector**  
   Use `jax.numpy.linalg.eigh` to find the eigenvalues and eigenvectors of the Hessian. A near-zero eigenvalue (e.g., $\lambda < 10^{-7}$) indicates a **valid direction of movement** (degree of freedom).  
   Extract the corresponding eigenvector $\mathbf{v}_0$.

3. **Take a small step in the eigenvector direction**  
   Move forward or backward based on desired motion direction:

   $$
   \mathbf{x}_{\text{new}} = \mathbf{x}_{\text{curr}} \pm \eta \mathbf{v}_0
   $$

   where $\eta$ is a step size hyperparameter.

4. **Reproject to the constraint manifold**  
   After stepping, we may drift off the constraint surface. We reproject using BFGS optimization to minimize the constraint loss.

5. **Detect locking**  
   Recompute the Hessian at the new configuration. If the smallest eigenvalue increases substantially (e.g., $\lambda > 10^{-3}$), we interpret this as a **locally rigid** configuration: a lock.

Implementation was done in JAX using automatic differentiation. Step sizes ranged from $\eta = 0.0001$ to $\eta = 0.75$ depending on the configuration and sensitivity.

---

## Stuck Linkage Results with Eigenvalues and Eigenvectors

### Freely Rotating vs Locked Linkage


![Eigenanalysis for stuck mechanism](/images/fully_locked_linkage_eigen.png)
![Eigenanalysis for freely rotating mechanism](/images/fully_rotating_linkage_eigen.png)

The **top chart** corresponds to a **locked** 4-bar mechanism. The **bottom chart** corresponds to a **freely rotating** mechanism.

Key observations:
- In the locked case, the smallest eigenvalue rapidly increases from near-zero to positive, indicating the **loss of a valid motion direction**. Once the smallest eigenvalue is in the positive range, the trajectory is quite unstable afterwards since we are taking steps in this positive eigenvalue direction despite already losing the valid motion direction. This can be further seen in the PCA analysis of the projected joint positions to 2 dimnesions where we do not get an oscillating line segment (partially locked) or a full circle (freely rotating) 
- In the free case, the zero eigenvalue **stays near zero**, confirming continuous motion along the constraint manifold.

---

### Partially Locked Linkage

![Eigenanalysis for partially stuck linkage](/images/partially_locked_linkage_eigen.png)

In the partially locked example above, the linkage has **minor room to wiggle**, but not enough for full rotation.

- The crank joint is at $(0,3)$ and the coupler joint is at $(0.1,3)$
- Compared to the fully locked case (e.g., crank at $(0,3)$ and coupler at $(10^{-5}, 3)$)

We observe:

- **Eigenvalue oscillations** around a small value (e.g., $(10^{-9})$)
- The mechanism **stays on a constrained path** — a 1D segment, not a loop

This suggests the eigenvalue trajectory can help differentiate **fully mobile**, **partially mobile**, and **fully locked** linkages. Additionally, by decreasing the step size in the bottom figure, the oscillatory behavior of the zero eigenvalue trajectory and the PCA analysis simply amplifies.

---

## Conclusion

By inspecting the **eigenstructure of the constraint Hessian**, we can detect linkage locking events from first principles. This analysis provides an algebraic foundation for:

- Lock-aware linkage design
- Neural network conditioning
- Reward/loss shaping in generative linkage models

If you'd like a deep dive into the code (written in JAX with BFGS reprojection and eigenanalysis), or to apply this to your own mechanisms, let me know!


## Acknowledgments
I would like to thank Alex Guerra, PhD candidate in the Department of Computer Science at Princeton University, and Ryan P. Adams, Professor of Computer Science at Princeton University, for their advising on this fun mini-project that was a part of my senior thesis (May 2025) at Princeton University.
