For each training step:
    
&nbsp;&nbsp;&nbsp;&nbsp;Generate full episode:
 - Collect **S = (s1, s2, .., sT)** (visited states)
 - Collect **A = (a1, a2, .., aT)** (actions taken) 
 - Collect **R = (r1, r2, .., rT)** (received rewards)
 
&nbsp;&nbsp;&nbsp;&nbsp;Calculate discounted reward for each step:
&nbsp;&nbsp;&nbsp;&nbsp;

$$\LaTeX$$
$$\int_\Omega \nabla u \cdot \nabla v~dx = \int_\Omega fv~dx$$