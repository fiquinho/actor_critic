For each training step:
    
&nbsp;&nbsp;&nbsp;&nbsp;Generate full episode:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Collect **S = (s1, s2, .., sT)** (visited states)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Collect **A = (a1, a2, .., aT)** (actions taken) 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Collect **R = (r1, r2, .., rT)** (received rewards)
 
&nbsp;&nbsp;&nbsp;&nbsp;Calculate discounted return for each step:

&nbsp;&nbsp;&nbsp;&nbsp;![Discounted_return](https://latex.codecogs.com/svg.image?g_{(s_{t}&space;,&space;a_{t})}&space;=&space;g_{t}&space;=&space;\sum_{k=0}^{T}&space;\gamma^{k}&space;*&space;r_{t&plus;k&plus;1}&space;\top&space;)

&nbsp;&nbsp;&nbsp;&nbsp;**G = (g1, g2, .., gT)**

