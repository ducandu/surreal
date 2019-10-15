# Surreal library and DeepGames.ai server.

### Incentive:
Design a novel python+tf2.0 based deep reinforcement learning library for 
"1d RL-algo coding | post-paper-comprehension".
This means the library allows users to code and implement any complex RL algorithm within 1 day, given that the paper 
describing the algorithm has been thoroughly understood.

This is accomplished by the following design choices:

##### Simplicity over speed:
Algo code looks very much like pseudocode and is extremely short. Our DQN2015 
algo is less than than 50 lines (not counting comments and imports). Same is true for all other algos.

  Example:
  
```python
    # Add sars't-tuple to memory.
    if self.s and self.a:
        self.memory.add_records(dict(s=self.s, a=self.a, r=r, s_=s_, t=t))  # s_ = s'; t = is s' terminal?

    # Handle ε-greedy exploration (decay).
    if random() < self.epsilon(time_percentage):
        a_ = self.a.sample()  # action/state-Spaces can be sampled
    else:
        a_ = np.argmax(self.Q(s_))  # looks like pseudocode: Q-function called directly
```

##### TensorFlow 2.0 based:
Tf2.0's eager execution mechanism by default allows for faster coding and better debuggability 
of our algos while maintaining the incredible speed we are used to from tensorflow 1.x.

##### Keras API:
Surreal's function approximators are implemented strictly following the Keras API and can
be directly called in various intuitive ways:
For example, a Q-function's constructor takes any Keras Model (or even a single Keras Layer) as its core network
and then builds the Q-function around it according to arbitrary action and state Spaces (yes, any deeply nested 
container Spaces are supported and automatically handled as well!). It then allows
for direct calling of this Q-function passing either only the state input
(all actions' Q-values are returned) or both state and action inputs (only the given action's single Q-value is 
returned).

```python
# A simple Q-network for some arbitrary action-space 
# (including complex container spaces likes deeply nested dicts and/or tuples).
Q = Network(network=tf.keras.layers.Dense(5), output_space=my_action_space)
q_vals = Q(s)  # <- returns q-values according to `my_action_space`.
q_val  = Q(s, a)  # <- returns the (single) q-value for action a.
```
##### Distribution-learning networks:
Equally easily, networks that learn distribution outputs can be constructed with Surreal. For example
for policy gradient algorithms, model based next-env-state predictions using Gaussian mixtures, or other typical 
deep-RL applications. Surreal also supports hybrid behaviors of networks outputting distributions AND actual values (as e.g. done in 
policy networks with shared value-function baselines).

```python
# A Policy network for any arbitrary action-space
# (including complex container spaces likes deeply nested dicts and/or tuples).
# Learns a distribution over the possible actions.
pi = Network(network=[{"name": "dense", "units": 256}], output_space=my_action_space, distributions=True)
a = pi(s)  # Sample an action given s (state).
a, log_likelihood = pi(s, log_likelihood=True)  # Sample an action given s and also return its log-likelihood.
likelihood = pi(s, a)  # Return the likelihood/prob of action given s.
```

##### Environment driven execution design ("the env runs the show"):
The environment controls execution in Surreal. Events are sent to the algorithms, which simply implement a ticking 
method, in which all algo execution logic is "stored". This tick method is the heart of the Algo and resembles the 
pseudocode found in the paper describing the algorithm.
In the future, this will allow for the integration of multi-agent-, distributed- and game-engine-driven execution 
patterns.

##### Mixing of execution logic and algorithmic (math) logic inside an RLAlgo class:
Many algorithms are characterized by their unique execution patterns. Thus, these two concepts 
(execution logic and mathematical logic) are intertwined and should not be separated artificially.

##### Strong reusability via well-tested, standard deep learning components:
Most algorithms share a large number of ever repeating, standard components such as memories, buffers, preprocessors,
function approximators, decay components, optimizers, multi-GPU handling components, "n-step" components, etc..
Surreal comes with all of these and they have been thoroughly tested. 50% of the code are test cases!

##### Memory-saving "next-record" logic in all memory components:
Memories for RL Algos (such as a replay buffer) usually waste lots of space due to the 
"next-state problem" (the next state s' must be tied to the state s in e.g. DQN's sars-tuples).
Surreal has a smart way of flexible and efficient "next-record" handling that even supports arbitrary 
n-step mechanics and thus saves up to half the memory space needed
(given the state-space is large/space-consuming, e.g. in Atari Envs). 
Running Atari experiments on a 24Gb machine with a memory that can keep 120k records is not a problem.


### Currently implemented algorithms (stay tuned for more):

- **DQN2015** (2015)
- **DDDQN** (double/dueling/n-step/prioritized-replay Q-learning)
- **SAC** (Soft Actor-Critic in its Feb 2019 version) for continuous as well as discrete action spaces.
- Coming up next: **DADS** (an auto option discovery algo)

