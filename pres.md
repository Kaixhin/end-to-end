## End-to-End Learning of Deep Visuomotor Policies

Kai Arulkumaran

24 April 2015

------------------

<iframe data-autoplay width="960" height="540" src="https://www.youtube.com/embed/Q4bMcUk6pcw" frameborder="0" allowfullscreen></iframe>

------------------

### Summary

- Policy search (reinforcement learning) to learn control for robot tasks
- Deep learning for learning features from low-level observations (joint angles + camera images) all the way to joint torques (no "computer vision system" or PD controller)

------------------

### Reinforcement Learning

- Agent in an environment learns through trial and error (no labels, just delayed signal)
- Set of states $X$ and set of actions $U$
- Policy $\pi$ determines action $u_t$ to perform given the state $x_t$
- Action transitions $x_t$ to $x_{t+1}$ with scalar reward $r_{t+1}$
- Maximise expected return (sum of rewards) $R$ given policy: $$E[R|\pi]$$

------------------

### Observability

- Markov Chain (MC) is a random process with state transitions (alternatively a random walk on a graph), with the Markov property: $$P[x_{t+1}|x_t] = P[x_{t+1}|x_t,...,x_0]$$
- Markov Decision Process (MDP) extends MC with actions and rewards
- **Full observability**: observations ($O$) give entire state of environment = MDP
- **Partial observability**: observations give incomplete state of environment = Partially Observable MDP (POMDP)

------------------

### Policy Search

- Parametrise the policy: $\pi_\theta(u_t|o_t)$
- Optimise performance of policy with objective function: $l(x_t, u_t)$
- Guided Policy Search
    1. Trajectory optimisation phase (fully observed)
    2. Supervised learning phase (partially observed)
- 1. Do not know dynamics but do know $x_t$ e.g. holding bottle in other hand - controlled environment training
- 2. Trained on $o_t$ to handle partial observability

------------------

### Partially Observed Guided Policy Search

- Trajectory: $\tau = \{x_1, u_1, ..., x_T, u_T\}$
- Trajectory phase produces Gaussian (mean) trajectory distributions: $p_i(\tau)$
- Succeed from several initial states (labels), final policy generalises from same distribution
- Outer loop samples trajectories from a policy
- Inner loop iteratively enforces agreement between $\pi_\theta(u_t|o_t)$ and $p_i(\tau)$

------------------

### Supervised Learning

- Possible training sets
    - Example demonstrations
    - Trajectory optimisation (known dynamics)
    - Trajectory-centric reinforcement learnt (unknown dynamics)

------------------

### Test 4

CNN used as function approximator for policy
$\pi_\theta$ is a Gaussian with mean from a nonlinear function approximator
Trained with objects in arm (full observability) but generalises to partial observability

------------------

### Weaknesses

- Large change in mass of target object
- Large change in visuals of objects/environment
- Pertubations that are too different from training
