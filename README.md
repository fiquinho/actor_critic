For each training step:
    
 - Generate full episode:
   - Collect **S = (s1, s2, .., sT)** (visited states)
   - Collect **A = (a1, a2, .., aT)** (actions taken) 
   - Collect **R = (r1, r2, .., rT)** (received rewards)
 
 - Calculate discounted return for each step:
   - ![Discounted_return](https://latex.codecogs.com/svg.image?g_{(s_{t}&space;,&space;a_{t})}&space;=&space;g_{t}&space;=&space;\sum_{k=0}^{T}&space;\gamma^{k}&space;*&space;r_{t&plus;k&plus;1})
   - **G = (g1, g2, .., gT)**


# Sources
 - http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf
 - https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
 - http://incompleteideas.net/sutton/book/RLbook2018.pdf
