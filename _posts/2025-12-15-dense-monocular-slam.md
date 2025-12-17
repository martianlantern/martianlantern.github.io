---
layout: post
title: Dense Monocular SLAM
date: 2025-12-15
comments: false
tags: [algorithm, optimization, 3d-reconstruction]
archive: false
---

SLAM aims to estimate the camera 3D poses as well as the structure of points in the scene over time. For vision based SLAM systems the input is a sequence of images from a single or multiple cameras. For numerical stability we also represent the 3d points using inverse depths i.e $$ (x, y, \rho) $$ where $$\rho=\frac{1}{Z}$$ here $$Z$$ is the depth and $$x, y$$ are normalized camera coordinates

The core idea of SLAM is bundle adjustment which is a non linear optimization problem that jointly optimizes the poses of each frame and the inverse depths of the points. It does this by minimizing the reprojection error or photometric error across all the frames and all the corresponding points. Let $$\xi_{i}$$ represent the 6-DoF pose parameter for camera $$i$$ in $$SE(3)$$ and $$d_{j}$$ denote the inverse depth for point $$j$$; also stack the pose and inverse depths in a state vector

$$
x=[\xi_{1}, ..., \xi_{N}, d_{1}, ..., d_{N}],
$$

then the cost function becomes 

$$
\min_{\{\xi_i, d_j\}} \sum_{i, j} \underbrace{\bigl\| \text{(observed pixel)} - \text{(predicted pixel from } \xi_i, d_j) \bigr\|^2}_{\text{error term}}.
$$

Due to the non linearity of the reprojection error we solve it by iterative methods (Gauss-Newton or Levenberg-Marquardt) by first linearizing the reprojection error around the current estimate with the Jacobian $J$ as $e(x+\Delta x) \approx e(x) + J\,\Delta x$ and solving the normal equations for the Hessian $$H$$ and getting an estimate of $$\Delta x$$

The cost function can now be written as

$$
\begin{align*}
E(x+\Delta x)&=\frac{1}{2}\|e(x+\Delta x)\|^2\approx\frac{1}{2}\|e(x)+J\,\Delta x\|^2 \\
E(x+\Delta x)&\approx\frac{1}{2}\Bigl(e(x)^Te(x)+2\,\Delta x^T\,J^T\,e(x)+\Delta x^T\,J^T\,J\,\Delta x\Bigr)
\end{align*}
$$

Minimizing this quadratic cost with respect to $$\Delta x$$ (by taking the derivative and setting it to zero) leads to:

$$
J^T\,J\,\Delta x=-J^T\,e(x)
$$

Defining the approximate Hessian $$H=J^T\,J$$ and $$b=-J^T\,e(x)$$, we reduce it down to

$$
H\Delta x=b
$$

The Hessian $$H$$ has a very well known block structure  

$$
H = 
\begin{bmatrix}
C & E \\
E^T & P
\end{bmatrix}, 
\quad
\mathbf{x} = 
\begin{bmatrix}
\Delta \xi \\
\Delta d
\end{bmatrix},
\quad
\mathbf{b} =
\begin{bmatrix}
\mathbf{v} \\
\mathbf{w}
\end{bmatrix}
$$

Here $$C$$ is a block diagonal or block sparse matrix representing the second derivatives of the cost function with respect to the camera poses alone, $$P$$ is also a sparse matrix representing the second derivatives with respect to the inverse depths of points, and $$E$$ and $$E^{T}$$ are cross terms that couple the camera parameters with the inverse depth parameters

We solve this by first eliminating $$\Delta d$$:

$$
\begin{align*}
\Delta d &= P^{-1}(w - E^{T}\Delta \xi) \\
C\Delta \xi + E\,P^{-1}(w - E^{T}\Delta \xi) &= v\\
\Bigl(C - E\,P^{-1}E^{T}\Bigr)\Delta \xi &= v - E\,P^{-1}w
\end{align*}
$$

The matrix $$C - E\,P^{-1}E^{T}$$ is also called the Schur complement of $$H$$ with respect to $$P$$ or $$H/P$$.

Once $$\Delta \xi$$ is computed from the equation above, the inverse depth update is recovered as:

$$
\Delta d = P^{-1}(w - E^{T}\Delta \xi).
$$

Finally, the camera poses and inverse depths are updated:  
- For each camera $$i$$, the pose update is performed using the pose composition in $$SE(3)$$:  
  $$
  \xi_i^{\,\text{new}} = \xi_i \oplus \Delta \xi_i,
  $$
  where $$\oplus$$ denotes the appropriate composition operator.  
- For each point $$j$$, the inverse depth update is given by: $d_j^{\,\text{new}} = d_j + \Delta d_j$