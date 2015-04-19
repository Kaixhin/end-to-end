## End-to-End Learning of Deep Visuomotor Policies

Kai Arulkumaran

24 April 2015

------------------

<iframe data-autoplay width="960" height="540" src="https://www.youtube.com/embed/Q4bMcUk6pcw" frameborder="0" allowfullscreen></iframe>

------------------

## Summary

- Policy search (reinforcement learning) to learn control for robot tasks
- Deep learning for learning features from low-level observations (joint angles + camera images) to joint torques

------------------

## Reinforcement Learning

- Learn through trial and error (no labels, just delayed signal)
- Set of states $S$ and set of actions $A$
- Policy $\pi$ determines action $a_t$ to perform given the state $s_t$
- Action transitions $s_t$ to $s_{t+1}$ with scalar reward $r_{t+1}$
- Maximise expected return (sum of rewards) $R$ given policy: $$\mathbb{E}[R|\pi]$$

------------------

## Observability

- Markov Chain (MC) is a random process with state transitions, has Markov property: $$\mathbb{P}[s_{t+1}|s_t] = \mathbb{P}[s_{t+1}|s_t,...,s_0]$$
- Markov Decision Process (MDP) extends MC with actions and rewards
- **Full observability**: observations ($O$) give entire state of environment = MDP
- **Partial observability**: observations give incomplete state of environment = Partially Observable MDP (POMDP)

------------------

## Policy Search

- Parametrise the policy: $$\pi_\theta(s, a) = \mathbb{P}[a|s,\theta]$$
- Optimise objective function that gives performance of policy
- Guided Policy Search
    - Trajectory optimisation with few conditions
    - Supervised learning from successful executions

------------------

## Test 4

Trained with objects in arm (full observability) but generalises to partial observability

------------------

## Weaknesses

- Large change in mass of target object
- Large change in visuals of objects/environment
- Pertubations that are too different from training
