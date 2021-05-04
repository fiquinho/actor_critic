For each training step:
    
&nbsp;&nbsp;&nbsp;&nbsp;Generate full episode:
 - Collect **S = (s1, s2, .., sT)** (visited states)
 - Collect **A = (a1, a2, .., aT)** (actions taken) 
 - Collect **R = (r1, r2, .., rT)** (received rewards)
 
&nbsp;&nbsp;&nbsp;&nbsp;Calculate discounted reward for each step:
&nbsp;&nbsp;&nbsp;&nbsp;

![Drag Racing](https://latex.codecogs.com/svg.image?G_{(s_{t}&space;,&space;a_{t})}&space;=&space;G_{t}&space;=&space;\sum_{k=0}^{T}&space;\gamma^{k}&space;*&space;r_{t&plus;k&plus;1}&space;)

