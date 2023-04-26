# Repeat_PINN
Working project


### Simulation Setting

#### Equation
First define the notation such that

$$\nabla^k_x u= \frac{\partial^k u(x,y)}{\partial x^k} , \nabla^k_y u= \frac{\partial^k u(x,y)}{\partial y^k}$$

and $\nabla^0 u = 2u(x,y)$. We define a high order nonlinear parametric PDE such that
$$
\begin{align*}
   &\mathcal{L}_\beta u= f(x,y) ,\quad x,y \in [0,10]^2\\
    & \nabla_x^{k}u,\nabla_y^ku= h_k(x,y),g_k(x,y),k=0,1,\cdots,5,\quad x,y\in \partial[0,10]^2
\end{align*}$$
where 
$$\begin{align*}
    \mathcal{L}_\beta =\beta_0 (\text{Id})^2+ \beta_1 (\nabla_x^1 \nabla^1_y) + \beta_2(\nabla_x^2+\nabla_y^2)^2+ \beta_3(\nabla_x^3+\nabla_y^3)+\beta_4(\nabla_x^4+\nabla_y^4)
\end{align*}$$
Choose 
$$\begin{align*}
    u(x,y) &= \frac{1}{\sqrt{5}}\sum_{k=1}^\infty (-1)^{k+1} k^{-(s+0.5)} (\cos (2k\pi x/10)+\sin (2k\pi x/10)) \\&+\frac{1}{\sqrt{5}}\sum_{k=1}^\infty 2 k^{-(s+0.5)}(\cos(2k\pi y/10)+\sin (2k\pi y/10))
\end{align*}$$
Using $\beta_i = 1$ as true parameter. Set noise as $N(0,0.1)$, smoothness $s=5$ and $576$ samples. 
