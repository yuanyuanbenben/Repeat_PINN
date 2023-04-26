# Repeat_PINN
Working project


### Simulation Setting

#### Equation
First define the notation such that

$$\nabla^k_x u= \frac{\partial^k u(x,y)}{\partial x^k} , \nabla^k_y u= \frac{\partial^k u(x,y)}{\partial y^k}$$

and $\nabla^0 u = 2u(x,y)$. We define a high order nonlinear parametric PDE such that

$$\mathcal{L}_\beta u= f(x,y) ,\quad x,y \in [0,10]^2,\quad \nabla_x^{k}u,\nabla_y^ku= h_k(x,y),g_k(x,y),k=0,1,\cdots,5,\quad x,y\in \partial[0,10]^2$$

where 

$$\mathcal{L}_\beta =\beta_0 (\frac{1}{2}\nabla^0)^2+ \beta_1 (\nabla_x^1 \nabla^1_y) + \beta_2(\nabla_x^2+\nabla_y^2)^2+ \beta_3(\nabla_x^3+\nabla_y^3)+\beta_4(\nabla_x^4+\nabla_y^4)$$

#### Solution and observation
Choose 

$$u(x,y) = \frac{1}{\sqrt{5}}\sum_{k=1}^\infty (-1)^{k+1} k^{-(s+0.5)} (\cos (2k\pi x/10)+\sin (2k\pi x/10)) +\frac{1}{\sqrt{5}}\sum_{k=1}^\infty 2 k^{-(s+0.5)}(\cos(2k\pi y/10)+\sin (2k\pi y/10))$$

Using $\beta_i = 1$ as true parameter. Set noise as $N(0,0.1)$, smoothness $s=5$ and $576$ samples. 

#### Results

         |Parameter            |$\beta_0$|$\beta_1$|$\beta_2$|$\beta_3$|$\beta_4|
         |:--------------------|:--------|:--------|:--------|:--------|:--------|
         |True value           | 1.000|1.000|1.000|1.000|1.000|
         |PINN                 | 0.983|  0.980| 1.147|  0.991|  0.881|
         |abs error$\times10^3$| 16.911| 19.983| 146.508|  9.215| 118.616|
         |One step modified |0.994|  0.998| 1.060|  1.015|  1.023|
         |abs error$\times10^3$ |5.762| 2.011| 59.631| 14.704| 23.081|
         |related error|0.341| 0.101| 0.407|1.596|  0.195|
         |Two step modified | 1.028 |1.000| 0.851| 1.014 |1.065|
         |abs error$\times 10^3$| 27.828|  0.233| 148.980| 13.616| 65.020|
         |related error ratio|1.645|  0.012| 1.017|  1.478|  0.548|
