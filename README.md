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
        self.memory.add_records(dict(s=self.s, a=self.a, r=r, s_=s_, t=t))

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
Surreal's function approximators are implemented strictly following Keras API standards and can
be directly called in various intuitive ways:
For example, a Q-function's constructor takes any Keras Model (or even a single keras Layer) as its core network
and then builds the Q-function around it according to arbitrary action and state Spaces (yes, any container Spaces are
supported and automatically handled). It then allows
for direct calling of this Q-function passing either only the state input
(all action Q-values are output) or both state and action inputs (only the given action's single Q-value is returned).

```python
# A simple Q-network for some arbitrary action-space 
# (including complex container spaces likes deeply nested dicts and/or tuples).
Q = Network(network=tf.keras.layers.Dense(5), output_space=my_action_space)
q_vals = Q(state)  # <- returns q-values according to `my_action_space`
q_val  = Q(state, some_action)  # <- returns the (single) q-value of `some_action`
```

##### Distribution-learning networks:
Equally easily, networks that learn distribution outputs can be constructed (e.g. for policy gradient
algorithms, model based next-env-state predictions using Gaussian mixtures, etc.). Surreal even
supports hybrid behaviors of networks outputting distributions AND actual values (as e.g. done in 
policy networks with shared value-function baselines).

```python
# A Policy network for any arbitrary action-space
# (including complex container spaces likes deeply nested dicts and/or tuples).
# Learns a distribution over the possible actions.
pi = Network(network=[{"name": "dense", "units": 256}], output_space=my_action_space, distributions=True)
action = pi(state)  # Sample an action given `state`.
action, log_likelihood = pi(state, log_likelihood=True)  # Sample an action given `state` and also return its log-likelihood.
likelihood = pi(state, action)  # Return the likelihood/prob of action given `state`.
```

##### Environment driven execution design ("the env runs the show"):
Events are sent to the algorithms by the env 
which are hence coded against a very simple interface. This will allow - in the future - for the 
integration of multi-agent-, distributed- and game-engine-driven execution patterns.

##### Mixing of execution logic and algorithmic (math) logic inside an RLAlgo class:
Many algorithms are characterized by their unique execution patterns. Thus, these two concepts 
(execution logic and mathematical logic) are intertwined and should not be separated artificially.

##### Strong reusability via ever repeating deep learning components:
Most algorithms share a large number of repeating, standard components such as memories, buffers, preprocessors,
function approximators, decay components, optimizers, multi-GPU handling components, "n-step" components, etc..
Surreal comes with all of these and they have been thoroughly tested. 50% of the code are test cases!



### Currently implemented algorithms (stay tuned for more):

- **DQN2015** (2015)
- **DDDQN** (double/dueling/n-step/prioritized-replay Q-learning)
- **SAC** (Soft Actor-Critic in its Feb 2019 version) for continuous as well as discrete action spaces.
- Coming up next: **DADS** (an auto option discovery algo)

