---
title: "Forward Linkage Kinematics with Conformal Geometric Algebra"
date: 2025-05-07
categories: [blog]
tags: [linkages, robotics, kinematics, geometric-algebra, Lie algebra, manifold-optimization, JAX]
layout: blog
comments: true
excerpt: "We present a novel formulation of planar closed-loop linkage kinematics using conformal geometric algebra (CGA), enabling compact, coordinate-free representation of linkage elements and constraints."
---

## Abstract

We present a novel formulation of planar closed-loop linkage kinematics using conformal geometric algebra (CGA), enabling compact, coordinate-free representation of linkage elements and corresponding constraints. Our approach leverages Lie algebra-based solvers to perform forward kinematic solving over multi-bar mechanisms, including 4-bar, 6-bar, and higher-order linkages. We demonstrate that encoding constraints directly in CGA multivector space improves geometric interpretability compared to traditional matrix or vector representations. We provide detailed evaluation across several linkage configurations, showcasing accurate motion tracking and constraint satisfaction even under complex multi-loop topologies.

## 1 Introduction

Planar linkage mechanisms consist of rigid bars constrained by joints and grounded to a reference frame. These linkages are fundamental to the field of robotics, mechanical design, and motion synthesis [Slocum, Thomaszewski et al., 2014, Zhang et al., 2010]. While traditional methods for simulating linkage kinematics rely on matrix algebra, coordinate parametrization, or vector-based solvers, they often struggle with representing geometric constraints compactly and handling multi-loop mechanisms robustly [Zaplana et al., 2022, Kleppe and Egeland, 2016]. Conformal geometric algebra (CGA) [Lundholm and Svensson, 2009] offers a unified language for encoding points, lines, circles, and rigid transformations, making it particularly attractive for modeling mechanical systems. In this paper, we introduce a CGA-based framework for forward kinematic simulation of planar linkages, incorporating a Lie algebraic solver [Etingof, 2024] to efficiently satisfy constraints and propagate motion.

## 2 Related Works

Planar linkage kinematics (our work focuses on closed kinematic loops) has seen interest from the mechanical engineering, robotics, and computational geometry communities. Early works rely on closed-form solutions for simple linkage designs, such as the handy four-bar linkage [de Jonge, 1954]. While closed-form solutions work quite well for 4-bar linkage mechanisms, they are often too difficult to solve for higher-order or multi-loop mechanisms [McCarthy, 2000].

At the same time, Lie group-based formulations arose to leverage the structure of SE(2) and SE(3) to parameterize rigid body motions [Gallo, 2023]. This allowed for more compact motion representations and improved numerical stability for robot kinematics and motion planning problems.

Recently, conformal geometric algebra is being used to model kinematics of mechanisms via robots with a special emphasis on inverse kinematics [Zaplana et al., 2022, Carbajal-Espinosa et al., 2024, Kleppe and Egeland, 2016]. Moreover, FIKA Carbajal-Espinosa et al. [2024] demonstrated real-time performance advantages with a CGA-based algorithm for inverse kinematics of a 7-DOF anthropomorphic arm. CGA was also applied to design specific chains with 3 revolute joints and geared mechanisms [Wang et al., 2022].

It is important to note that forward linkage kinematics for complex, closed-loop linkages focuses on solving a system of nonlinear equations over time. These nonlinear equations increase in quantity with multiple loops or when more degrees of freedom are introduced in the system.

The advantages of our approach is that by developing a forward kinematics solver for general planar linkages using CGA, combined with Lie algebra-based solvers, we can effectively encode constraints compactly and uniformly with CGA multivectors. This representation avoids coordinate specific representations. CGA handles points, lines, and circles very easily, allowing flexible and geometrically meaningful modeling of revolute and prismatic joints. The use of Lie algebras allow us to incremently update rotors via exponential maps of nudged bivectors into the corresponding Lie group element, enabling efficient numerical solving and ensuring we don’t have to add link length constraints since our Lie algebra preserves link lengths in rigid body transformations. Equally important, we can stay entirely in CGA without explicit projections, enhancing interpretability of our kinematics solver. While prior works in CGA predominantly focus on inverse kinematics or single-mechanism configurations, we develop a scalable forward kinematics framework that allows for effective mechanism kinematics simulation over various complexities.

## 3 Methods

### 3.1 Geometric Algebra Representation

We model planar linkage mechanisms using conformal geometric algebra (CGA), embedding 2D Euclidean points into a higher-dimensional algebra to represent points, lines, and rigid-body motions in a unified framework.

Each link is represented as a line segment between two conformal points:

$$
X_{\text{ref}} = \{x_1, x_2\}, \quad x_i \in \mathbb{R}^{n,1,1}
$$

where each $x_i$ is a null vector in CGA. We transform these points into the world frame at time $t$ via the sandwich product:

$$
X_{\text{world}} = R X_{\text{ref}} \tilde{R}
$$

where
- $R$ is the rotor encoding $SE(2)$ (special Euclidean group 2 Lie group) motion,
- $\tilde{R}$ is the reverse of $R$,
- $X_{\text{ref}}$ is the set of points in the local (body) frame.


Points $p \in \mathbb{R}^2$ are embedded into CGA via

$$
x = \text{up}(p) = p + \frac{1}{2} \|p\|^2 e_\infty + e_0
$$

### 3.2 Constraint Formulation

We enforce linkage closure constraints symbolically:

- **Rigid bar constraint**: for connected joints $(x_i, x_j)$,

$$
C_{ij} = x_i \wedge x_j = 0
$$

- **Pinning constraint**: to fix a joint at a point,

$$
C_{\text{pin}} = x_j \wedge x_{\text{fixed}} = 0
$$

- **Sliding constraint (prismatic)**:

$$
C_{\text{slide}} = x_i \wedge x_j \wedge n = 0
$$

where $n$ encodes the sliding line’s direction.

### 3.3 Lie Algebra Linearization and Solver

We use the Lie algebra of $SE(2)$, with basis:

$$
\mathfrak{se}(2) = \left\{ \alpha(e_1 \wedge e_2) + \beta(e_1 \wedge e_0) + \gamma(e_2 \wedge e_0) \right\}
$$

to parametrize small updates as bivectors $B$. The rotor update at each timestep is:

$$
R(t) = \exp(B) R(t-1)
$$

At each Newton iteration, we solve:

$$
J \Delta B = -F(B)
$$

where:

- $F(B)$ stacks the residuals of the constraints,
- $J = \frac{\partial F}{\partial B}$ is the Jacobian,
- $\Delta B$ is the Lie algebra step direction.

We compute $J$ efficiently using automatic differentiation (JAX’s `jvp`).

### 3.4 Constraint-driven Time Stepping

For each timestep:

$$
\begin{cases}
R_{\text{crank}}^{(t)} = \exp(t \theta (e_1 \wedge e_2)) R_{\text{crank}}^{(t-1)} \\
R_{\text{other}}^{(t)} = \text{Newton solve: } \min_{\Delta B} \|F(\Delta B)\|
\end{cases}
$$

This ensures that after advancing the crank, all other links are updated by solving for feasible configurations that satisfy the closure constraints. Advantages of our approach include a fully symbolic coordinate-free representation, joint handling of rotations and translations via rotors, fast jacobian computation through automatic differentiation in JAX, and high-extensibility since new linkage types just require additional constraints to the constraint function.

## 4 Results

In the below static images of linkage kinematics simulations in Figure 1, the green bar represents the crank (driving) link and the blue bar represents the ground link.

<img src="/images/four_bar_rev.png" alt="4-bar linkage" style="width: 400px; height: 300px; object-fit: contain;" />  
*(a) 4-bar linkage mechanism with 4 revolute joints, 1 ground link, and 1 degree of freedom.*

<img src="/images/four_bar_prismatic.png" alt="4-bar prismatic linkage" style="width: 400px; height: 300px; object-fit: contain;" />  
*(b) 4-bar linkage mechanism with 3 revolute joints, 1 prismatic joint, 1 ground link, and 1 degree of freedom. The prismatic joint traces the vertical line between (0,3) and (0,5).*

<img src="/images/sixbar.png" alt="6-bar linkage" style="width: 400px; height: 300px; object-fit: contain;" />  
*(c) 6-bar linkage mechanism with 5 revolute joints, 1 ground link, and 1 degree of freedom.*

<img src="/images/ten_bar.png" alt="10-bar linkage" style="width: 400px; height: 300px; object-fit: contain;" />  
*(d) 10-bar linkage mechanism with 7 revolute joints, 1 ground link, and 1 degree of freedom.*

Our results can easily be extended to multiple degree of freedom linkage designs where we maintain 2 separate crank rotors. These experiments are currently ongoing and are not included in this version of the paper.

## 5 Conclusion

We present a robust framework for forward kinematic simulation of closed-loop planar linkages using conformal geometric algebra. By encoding constraints as multivector residuals and leveraging Lie algebra solvers, we achieve numerically stable, geometrically meaningful simulations that generalize across linkage complexities. Our approach highlights the power of conformal geometric algebra in mechanical modeling, offering a coordinate-free, compact representation that scales to multi-loop systems. While the paper provides a single step of the kinematic simulation, all full kinematics simulations for all linkages can be found in the public google drive folder linked here:

[Simulation folder link](https://drive.google.com/drive/folders/1T6v5Bz9mVF48Ef1GQZOQ5ZObtqV_QTQW?usp=share_link)

Current work has linkage kinematics simulations running on the CPU, shifting kinematics simulations to GPUs can easily allow us to vectorize our constraint function (`vmap`) across linkage mechanisms varying in complexity regardless of the constraints (revolute joints, prismatic joints, ground links, etc.) since they are all based in the wedge product.

## 6 Future Work

Future directions will include expanding kinematics to other types of mechanisms, such as open-chain linkages, spur gears, cams, and more. We would like to also analyze the singularities of linkage mechanisms in CGA by analyzing the Jacobian of the constraint function. We only need to analyze the Jacobian and not the Hessian because our constraint does not output a single scalar value. This would first involve a stronger understanding of Lie derivatives and incorporating them in our current solvers.

## 7 Acknowledgments

I would like to thank Alex Guerra, PhD candidate in the Department of Computer Science at Princeton University for his advising and making important contributions in developing a geometric algebra library in JAX, including implementing lie algebras in geometric algebra, which this project uses. I would like to also thank Ryan P. Adams, Professor of Computer Science at Princeton University, for his advising on this project. This work was a part of my senior thesis (May 2025) at Princeton University.

## 8 References

- [Oscar Carbajal-Espinosa, Leobardo Campos-Macías, and Miriam Díaz-Rodriguez. "FIKA: A Conformal Geometric Algebra Approach to a Fast Inverse Kinematics Algorithm for an Anthropomorphic Robotic Arm." *Machines*, 12(1), 2024.](https://www.mdpi.com/2075-1702/12/1/78) DOI: [10.3390/machines12010078](https://doi.org/10.3390/machines12010078)

- [A. E. R. de Jonge. "Discussion: An Analytical Approach to the Design of Four-Link Mechanisms" (Freudenstein, 1954). *ASME Transactions*, 76(3):489–490, 1954.](https://doi.org/10.1115/1.4014882) DOI: [10.1115/1.4014882](https://doi.org/10.1115/1.4014882)

- [Pavel Etingof. "Lie Groups and Lie Algebras", 2024. *arXiv preprint* arXiv:2201.09397.](https://arxiv.org/abs/2201.09397)

- [Eduardo Gallo. "The SO(3) and SE(3) Lie Algebras of Rigid Body Rotations and Motions and Their Application to Discrete Integration, Gradient Descent Optimization, and State Estimation", 2023. *arXiv preprint* arXiv:2205.12572.](https://arxiv.org/abs/2205.12572)

- [Adam L. Kleppe and Olav Egeland. "Inverse Kinematics for Industrial Robots Using Conformal Geometric Algebra." *Modeling, Identification and Control*, 37(1):63–75, 2016.](https://www.mic-journal.no/ABS/MIC-2016-1-6.asp) DOI: [10.4173/mic.2016.1.6](https://doi.org/10.4173/mic.2016.1.6)

- [Douglas Lundholm and Lars Svensson. "Clifford Algebra, Geometric Algebra, and Applications", 2009. *arXiv preprint* arXiv:0907.5356.](https://arxiv.org/abs/0907.5356)

- [Michael McCarthy. *Geometric Design of Linkages*. Interdisciplinary Applied Mathematics. Springer, New York, NY, 1st ed., 2000.](https://doi.org/10.1007/b98861) ISBN: 978-0-387-22735-1

- [Alexander Slocum. *Fundamentals of Design, Topic 4: Linkages*. Lecture Slides from MIT Course on Fundamentals of Design, 2008.](https://meddevdesign.mit.edu/wp-content/uploads/simple-file-list/FUNdaMentals-Chapters/FUNdaMENTALs-Topic-4.pdf)

- [Bernhard Thomaszewski, Stelian Coros, Damien Gauge, Vittorio Megaro, Eitan Grinspun, and Markus Gross. "Computational Design of Linkage-Based Characters." *ACM Transactions on Graphics (TOG)*, 33(4):64:1–64:9, 2014.](https://doi.org/10.1145/2601097.2601132)

- [L. Wang, G. Yu, L. Sun, Y. Zhou, and C. Wu. "Motion Generation of a Planar 3R Serial Chain Based on Conformal Geometric Algebra with Applications to Planar Linkages." *Mechanical Sciences*, 13(1):275–290, 2022.](https://ms.copernicus.org/articles/13/275/2022/) DOI: [10.5194/ms-13-275-2022](https://doi.org/10.5194/ms-13-275-2022)

- [Isiah Zaplana, Hugo Hadfield, and Joan Lasenby. "Closed-Form Solutions for the Inverse Kinematics of Serial Robots Using Conformal Geometric Algebra." *Mechanism and Machine Theory*, 173:104835, 2022.](https://doi.org/10.1016/j.mechmachtheory.2022.104835)

- [Yi Zhang, Susan Finger, and Stephannie Behrens. *Introduction to Mechanisms*. Carnegie Mellon University, 2010.](https://www.cmu.edu/me/robomechanics/design/IntroMechanisms/index.html)
