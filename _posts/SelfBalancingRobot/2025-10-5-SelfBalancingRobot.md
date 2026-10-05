---
layout: post
title: "Self-Balancing Robot: Inverted Pendulum"
tag: [Electronics, Control]
---

<link href="/css/syntax.css" rel="stylesheet">


## Motivation
In this post, I will summarize my adventure building a segway-type self-balancing robot. 

The initial idea was to do system identification; I wanted to find a linearization around the unstable upright equilibrium and the mapping from input ([PWM](https://en.wikipedia.org/wiki/Pulse-width_modulation)) to state. And I wanted to do so in a data-driven manner! Eventually I wanted to construct an [LQR](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator) controller. That turned out to be wildly difficult. I will not get into too much detail...  The original idea was to construct $X_1$, $X_2$ and $U$ data matrices and solve a least squares problem:

$\text{minimize}_{A, B} \| X_2 - \begin{bmatrix}A & B \end{bmatrix}\begin{bmatrix}X_1 \\\ U \end{bmatrix} \| ^2_F$



Turns out, perhaps not surprisingly, that estimating that matrix from sensor data while actively stabilizing the robot with [PID](https://en.wikipedia.org/wiki/PID_controller) is not easy. And, yeah... the equilibrium is unstable, so to find the linearization with my naive approach, you first have to stabilize it. There are likely better ways, but that was my naive idea as a beginner! 

The first thing that you see is that, indeed, PD input is a linear combination of the state. So $\begin{bmatrix}X_1 \\\ U \end{bmatrix}$ is rank deficient, kind of by definition. So you need to stabilize and perturb the system at the same time. Getting this right is hard, especially when working with sensor data. 

In a way, what I am presenting here is work half-done. I am showing just PID (actually PD), like everybody else. I gave up on trying to identify the system... for now. It is quite interesting, but it requires more study first. Regardless, the combination of hardware and electronics with software makes this still a very challenging (and rewarding) project. 

Despite not using an LQR, it will be useful to clarify the state of the system. At any time point $t$, the state is defined as:

$$\begin{bmatrix}  \theta \\\ \dot{\theta} \\\ x \\\ \dot{x} \end{bmatrix}$$

Where $\theta$ is the angle with respect to the perpendicular to the ground and $x$ is the distance with respect the initial position where we put the robot (so we want this to be $\overrightarrow{0}$). At first glance, $x$ seems way less important the $\theta$. However, since the motors of the robot saturate (the force they can exert diminishes at higher speeds), and I do not have an infinite room, it is actually pretty important to force the robot to stay close to $x=0$ and is actually crucial to avoid high speeds $\dot{x}$.


## Pieces
Hardware used:
- Microcontroller: ESP32-32U (I bought the antenna separately)
- Motor driver: TB6612 DRV8833
- Accelerometer + Gyroscope: BMI160
- Wheels: 7.4V with quadrature encoder included
- Battery: Two lithium 3.7V batteries, 3500mAh each
- General electronics hobby stuff: perfboard, cable, buck converter...

## Pictures
As you might have guessed the objective is to keep this upright!

<p align="center" style="display: flex; justify-content: center; gap: 20px;">
  <img src="/assets/images/SelfBalancingRobot/SBR.jpeg" alt="front" width="400"/>
  <img src="/assets/images/SelfBalancingRobot/SBR1.jpeg" alt="back" width="400"/>
</p>


## Sensor data handling.
Sensor data from the accelerometer is easily transformed into an angle; the gravity force split between the z and y (or x) axis gives us $\cos(\theta)$ and $\sin(\theta)$ respectively. However, this is quite noisy. To have a reliable estimation of the state, I used a Kalman Filter (KF). I must mention, however, that the much simpler complementary filter works also well. A KF provides the optimum state estimation with a weighted sum of the data coming from the sensor and the "predicted" state based on our model. If you have seen a bit of Bayesian Statistics, this is nothing else than conjugate updating Gaussian-Gaussian and a recursive relationship (+ linear algebra). The fundamental idea is that, at each point, a model proposes a prior of the current state based on the previous state. With that prior, and the likelihood constructed with the current sensor measurement, a new Gaussian is constructed. The mode of that Gaussian is the current estimation of the state. I am not going to get into the details of this here (the linear algebra part is actually tricky). The key idea is that the KF smooths the sensor data using information of what is physically plausible.

We do not have the equations of motion. That would be the gold standard to construct the KF. However, we can still use basic physics to construct a simple model for the KF. In this case, I use a model that simply says: current angle is the sum of the angular velocity, but mind the drift:

$$
\begin{align*}
\begin{bmatrix} \theta \\ b \end{bmatrix}_k &= \begin{bmatrix} 1 & -\Delta t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} \theta \\ b \end{bmatrix}_{k-1} + \begin{bmatrix} \Delta t \\ 0 \end{bmatrix} \omega_k.
\end{align*}
$$

Note that the KF state has two dimensions. Where $b$ is the bias/drift of the gyroscope and $w_t$ is the angular speed. So at every time point, the accelerometer information is somoothed out using the gyroscope data and this model. An interesting part of this is that it readily provides a clean derivative $\dot{\theta}$ from the gyroscope reading, since we are estimating that bias. In any case, there are parameters to tune here regarding how much we trust the model vs the sensor data.

The rest of the state: position and speed, comes from the quadrature encoders. Using a dedicated core of the ESP32 (it has 2) to handling this allows to have hardware interrupts to not miss any tick. In that sense, this sensor data is clean and goes directly into the state. Without hardware interrupts the estimations of position and hence speed would not be reliable.

Alright! So we've got ourselves a state. Now we want to act on the robot through the voltage provided to the wheels (PWM) and drive that state to $\overrightarrow{0}$.

## Control algorithm
The basic idea of this kind of self-balancing robot is to catch the fall. Within the PID framework, this simply boils down to applying a force proportional to the angle with respect to the normal to the ground (or some slightly different angle if the center of mass is not perfectly positioned!). In particular, PWM (in either direction, it is an integer from 0 to 255) will be applied to the wheels so that they move in the direction of the tilt. This is the P(roportional) part of the PID. With that alone, the control overshoots. Since it only really stops correcting at 0 error, it might have a lot of momentum in the opposite direction. To smooth this out, we use the derivative (D in PID) of the angle, given to us by the gyroscope. In the recovery phase, the sign of the derivative changes with respect to that of the error, diminishing the total PWM we use. I did not use the I(ntegral) part, so I am using PD control.

The main issue is to find the P and D constants. I resorted to trial and error in a pretty chaotic way, which is probably the worst thing you can do. After many attempts, I concluded this was not going to work. What I observed was the following: The robot sort of stabilizes, but at some point it tilts and never recovers. It moves forward accelerating, trying to correct, until the motors just saturate or it hits a wall. 

The problem here was that, under this simple formulation, the robot does not know where it is or at which speed it moves. This generates a situation in which the tilt is not quite enough to cause an overshoot but enough to drive the robot in a particular direction, so it keeps accelerating. In principle, a properly tuned PID algorithm could make this work but I did not manage.

Indeed, turns out that encoder ($x$ and $\cdot{x}$) information is actually pretty important if you want to make this work (and make your life easier). To use this information I use a "cascaded PID". The (average over the wheels) position and speed is used through another PD controller to propose a deviation of the target angle up to 3.5 degrees (another bit to tune). If the speed is high to the left, it will increase the target angle in the opposite direction, effectively reducing the force in that direction. This seems to be enough to fix the runaway problem, there is no high speed at low angles anymore. However, it does not work perfect to prevent some minor drift.

Now, the problem is that there are 2 loops, 4 variables (ignoring I in both)... tuning this by hand is horrible. If you are going to do something like that, be procedural. And putting some sticks on the robot to prevent it falling beyond 20 degrees might help!

So, that's the idea. We've got our state, we've got a way of keeping the state around the equilibrium... let's see how this does!


## Video: Balancing + State streaming
<video src="/assets/images/SelfBalancingRobot/Merged.mp4" controls width="100%">
  Your browser does not support the video tag.
</video>

It can balance for a long time, although it some situations it still crashes. It is still work in progress, I will revisit this at some point.

## Code
For details, check out the code [HERE](https://github.com/JordiAlonsoEsteve/SelfBalancingSegway/tree/main)

