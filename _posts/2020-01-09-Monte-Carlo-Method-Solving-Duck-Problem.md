---
title: 'Using Monte-Carlo Simulation to solve an interesting duck problem.'
subtitle: 'Intensity Surfaces and Gradients'
date: 2020-01-09
description: 
featured_image: '/images/blogs/01-09-20-Monte_Carlo/mc_banner.png'
---

<img src="/images/blogs/01-09-20-Monte_Carlo/1521260.png" alt="drawing" width="150"/>

# Introduction

Back to the end of 2019 when I was traveling back to China, I saw an very cute and interesting problem on my Moments (a social network on WeChat). The problem is described in the picture below: 

<img src="/images/blogs/01-09-20-Monte_Carlo/WechatIMG30.jpeg" alt="drawing" width="400"/>

## Problem Definition

The basic description of this problem is this: 
> Suppose I have four ducks randomly located inside a round pool, what is the probability that the four ducks happens to be in the same semi-circle? 

The first time I see this question, my instant intuition is to simulate it using the monte-carlo simulation. And the results for the simulation does converge to the actual probability. I'll show you how I achieve this using the code below. 

First, we'll do some import statement. The libraries we're using are pretty simple and common, they are ```random```, ```math``` and ```numpy```. 

```python
from random import random, uniform
from math import sqrt 
import numpy as np
```

The intuition I have is to directly randomly generate 4 points (ducks) inside the round circle (pool) for a large amount of times, and to see the probability or ratio between the groups that are in the same semi-circle and the total number of groups I sampled. 
Following the intuition, we need some functions to determine multiple conditions: 

1. **How to determine if a point (duck) is inside the circle (pool) or not.**
2. **How to determine if four points are inside the same semi-circle.**
3. **Done simulation in a more efficient and pythonic way.**

After solving the above requirements/conditions, we're good to simulate. 

------

# Monte Carlo Simulation Explained

## Function "```inside_circle```" for Condition 01. 
The function ```inside_circle``` basically takes the input of the coordinates of both randomly assigned point, the center of the circle (pool) as well as its radius and returns the decision. 
The algorithm is fairly simple, as it only computes the distance of both centers using the distance formula and compares it to the radius. If the distance is greater than radius, we conclude the simulated point is outside of the pool (returns ```false```) and vice-versa (returns ```true```). 

Distance Formula: 

$$d = \sqrt{(x_1 - x_2)^2 + (y_1-y_2)^2}$$

```python
def inside_circle(center_coord, curr_coord, radius): 
    return sqrt((curr_coord[0]-center_coord[0])**2 + (curr_coord[1]-center_coord[1])**2) < radius
```

## Function "```semicircle```" for Condition 02. 
The function ```semicircle``` serves the purpose of determining if four points are in the same semi-circle or not. As this is the *core* function for the monte-carlo simulation, it's a bit more complicated comparing to other functions. 

The general idea for this algorithm is that, we first connects one point to the circle so that we draw a line that go through the center, thus created a semi-circle inside the pool. And next, we calculate the function of that line as well as the sign of the rest of points respect to the line. If the signs are consistent, means all positive or all negative, we conclude that the four ducks are in the same semi-circle. And if the signs are not consistent, we draw a line between next point and the center, until all points are used. 

```python
def semicircle(center_coord, curr_row): 
    """
    @param center_coord:    The coordinate of pool's center, represented as a tuple. 
    @param curr_row:        The coordinate of four simulated ducks (points), represented as a list of tuples. 
    @returns:               True if four points are in the same semi-circle, false if not. 
    """
    # Calculate the line function for each coord and center. 
    # Ax + By + C = 0: A=y2-y1, B=x1-x2, C=x2*y1-x1*y2. 2: center, 1: current point
    result_list = []
    for i in range(4): 
        A = center_coord[1]-curr_row[i][1]
        B = curr_row[i][0]-center_coord[0]
        C = center_coord[0]*curr_row[i][1]-curr_row[i][0]*center_coord[1]
        semicircle_result = [A*curr_row[j][0]+B*curr_row[j][1]+C >= 0 for j in range(4) if j != i]
        if len(set(semicircle_result)) == 1: 
            return True
    return False
```

Since the inputs and loops for this function are fixed (4 for the for loop and 4 for the semicircle list comprehension), the time complexity is $O(n)$

Updated at feb 15, 2020, 
to be continued...

```python
def main(canvas_size, sim_size): 
    canvas_radius = float(canvas_size/2)
    center_coord = (canvas_radius, canvas_radius)

    curr_sim_size = 0
    curr_success_size = 0
    curr_sim_matx = np.empty((sim_size+1, 5))
    total_sim_count = 0

    while curr_sim_size <= sim_size: 
        total_sim_count += 1

        temp_row = [(uniform(0, canvas_size), uniform(0, canvas_size)) for i in range(4)]
        temp_row_truth = [inside_circle(center_coord, temp_coord, canvas_radius) for temp_coord in temp_row]

        if temp_row_truth != [1,1,1,1]: # Make sure the sampled result is within the circle. 
            continue
        else: 
            result_semicircle = semicircle(center_coord, temp_row)
            if result_semicircle: 
                curr_success_size += 1
            curr_sim_matx[curr_sim_size] = temp_row.append(result_semicircle)
            curr_sim_size += 1
    
    return float(curr_success_size / sim_size)
            

if __name__ == "__main__": 
    print(main(50, 1000000))
```