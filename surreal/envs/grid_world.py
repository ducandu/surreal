# Copyright 2019 ducandu GmbH, All Rights Reserved
# (this is a modified version of the Apache 2.0 licensed RLgraph file of the same name).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import numpy as np
import os
import random
import time

from surreal.spaces import Int, Bool, Dict
from surreal.envs.single_actor_env import SingleActorEnv


# Init pygame?
pygame = None
try:
    import pygame
    # Only use pygame if a display is available.
    pygame.display.init()

except ImportError:
    print("PyGame not installed. No human rendering possible.")
    pygame = None
except pygame.error:
    print("No display for PyGame available. No human rendering possible.")
    pygame = None


class GridWorld(SingleActorEnv):
    """
    A classic grid world.

    Possible action core are:
    - up, down, left, right
    - forward/halt/backward + turn left/right/no-turn + jump (or not)

    The state space is discrete.

    Field types are:
    'S' : starting point
    ' ' : free space
    'W' : wall (blocks, but can be jumped)
    'H' : hole (terminates episode) (to be replaced by W in save-mode)
    'F' : fire (usually causing negative reward, but can be jumped)
    'G' : goal state (terminates episode)

    TODO: Create an option to introduce a continuous action space.
    """
    # Some built-in maps.
    MAPS = {
        "chain": [
            "G    S  F G"
        ],
        "long-chain": [
            "                                 S                                 G"
        ],
        "2x2": [
            "SH",
            " G"
        ],
        "4x4": [
            "S   ",
            " H H",
            "   H",
            "H  G"
        ],
        "8x8": [
            "S       ",
            "        ",
            "   H    ",
            "     H  ",
            "   H    ",
            " HH   H ",
            " H  H H ",
            "   H   G"
        ],
        "4-room": [  # 30=start state, 79=goal state
            "     W     ",
            " H   W     ",
            "        G  ",
            "     W     ",
            "     W     ",
            "W WWWW     ",
            "     WWW WW",
            "     W F   ",
            "  S  W     ",
            "           ",
            "     W     "
        ]
    }

    def __init__(self, world="2x2", actors=None, save_mode=False, action_type="udlr",
                 reward_function="sparse", state_representation="discrete"):
        """
        Args:
            world (Union[str,List[str]]): Either a string to map into `MAPS` or a list of strings describing the rows
                of the world (e.g. ["S ", " G"] for a two-row/two-column world with start and goal state).

            save_mode (bool): Whether to replace holes (H) with walls (W). Default: False.

            action_type (str): Which action space to use. Chose between "udlr" (up, down, left, right), which is a
                discrete action space and "ftj" (forward + turn + jump), which is a container multi-discrete
                action space. "ftjb" is the same as "ftj", except that sub-action "jump" is a boolean.

            reward_function (str): One of
                sparse: hole=-5, fire=-3, goal=1, all other steps=-0.1
                rich: hole=-100, fire=-10, goal=50, all other steps=-0.1

            state_representation (str):
                - "discrete": An int representing the field on the grid, 0 meaning the upper left field, 1 the one
                    below, etc..
                - "xy": The x and y grid position tuple.
                - "xy+orientation": The x and y grid position tuple plus the orientation (if any) as tuple of 2 values
                    of the actor.
                - "camera": A 3-channel image where each field in the grid-world is one pixel and the 3 channels are
                    used to indicate different items in the scene (walls, holes, the actor, etc..).
        """
        # Build our map.
        if isinstance(world, str):
            self.description = world
            world = self.MAPS[world]
        else:
            self.description = "custom-map"

        world = np.array(list(map(list, world)))
        # Apply safety switch.
        world[world == 'H'] = ("H" if not save_mode else "F")

        # `world` is a list of lists that needs to be indexed using y/x pairs (first row, then column).
        #self.worlds = [world]
        self.n_row, self.n_col = world.shape

        # Figure out our state space.
        assert state_representation in ["discrete", "xy", "xy+orientation", "camera"]
        self.state_representation = state_representation
        # Discrete states (single int from 0 to n).
        if self.state_representation == "discrete":
            state_space = Int(self.n_row * self.n_col)
        # x/y position (2 ints).
        elif self.state_representation == "xy":
            state_space = Int(low=(0, 0), high=(self.n_col, self.n_row), shape=(2,))
        # x/y position + orientation (3 ints).
        elif self.state_representation == "xy+orientation":
            state_space = Int(low=(0, 0, 0, 0), high=(self.n_col, self.n_row, 1, 1))
        # Camera outputting a 2D color image of the world.
        else:
            state_space = Int(0, 255, shape=(self.n_row, self.n_col, 3))

        # Specify the actual action space.
        self.action_type = action_type
        action_space = Int(4) if self.action_type == "udlr" else Dict(dict(
            forward=Int(3), turn=Int(3), jump=(Int(2) if self.action_type == "ftj" else Bool())
        ))
        # Call super.
        super(GridWorld, self).__init__(actors=actors, state_space=state_space, action_space=action_space)

        # Define all n worlds.
        self.worlds = np.array([world] * len(self.actors))

        # Store the goal position for proximity calculations (for "potential" reward function).
        (self.goal_y,), (self.goal_x,) = np.nonzero(world == "G")

        assert reward_function in ["sparse", "rich"]  # TODO: "potential"-based reward
        self.reward_function = reward_function

        (start_y,), (start_x,) = np.nonzero(world == "S")
        # The default starting position (as  single discrete (int) state).
        self.default_start_pos = self._get_discrete_pos(start_x, start_y)

        # The actual observation (according to `state_representation`).
        self.state = np.array([state_space.zeros()] * len(self.actors))
        # The current discrete (int) positions of each actor.
        self.discrete_pos = np.array([self.default_start_pos for _ in self.actors], dtype=np.int32)
        # Only used if `state_representation`=='xy+orientation'. Int: 0, 90, 180, 270
        self.orientations = np.zeros(shape=(len(self.actors),), dtype=np.int32)
        # Only used, if `state_representation`=='cam'.
        self.cam_pixels = np.zeros(shape=(len(self.actors), self.n_row, self.n_col, 3), dtype=np.int32)

        # Reset ourselves.
        for i in range(len(self.actors)):
            self.state[i] = self._reset(actor_slot=i, randomize=False)

        # Init pygame (if installed) for visualizations.
        if pygame is not None:
            self.pygame_field_size = 30
            pygame.init()
            self.pygame_agent = pygame.image.load(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "images/agent.png")
            )
            # Create basic grid Surface for reusage.
            self.pygame_basic_surface = self._grid_to_surface()
            self.pygame_display_set = False

    @property
    def x(self):
        """
        Returns:
            np.ndarray[int]: The per-Actor array of current x-positions.
        """
        return self._get_x_y(self.discrete_pos)[0]

    @property
    def y(self):
        """
        Returns:
            np.ndarray[int]: The per-Actor array of current y-positions.
        """
        return self._get_x_y(self.discrete_pos)[1]

    def seed(self, seed=None):
        if seed is None:
            seed = time.time()
        np.random.seed(seed)
        return seed

    def _reset(self, actor_slot, randomize=False):
        """
        Args:
            randomize (bool): Whether to start the new episode in a random position (instead of "S").
                This could be an empty space (" "), the default start ("S") or a fire field ("F").

        Returns:
            any: The (single Actor/non-batched) state after the reset.
        """
        if randomize is False:
            self.discrete_pos[actor_slot] = self.default_start_pos
        else:
            # Move to a random first position (" ", "S", or "F" (ouch!) are all ok to start in).
            while True:
                self.discrete_pos[actor_slot] = random.choice(range(self.n_row * self.n_col))
                if self.worlds[actor_slot, self.y, self.x] in [" ", "S", "F"]:
                    break

        self.orientations[actor_slot] = 0

        return self._refresh_state(actor_slot)

    def _act(self, actions):  #, set_discrete_pos=None):
        """
        Action map:
        0: up
        1: right
        2: down
        3: left

        Args:
            actions (Optional[int,Dict[str,int]]):
                For "udlr": An integer 0-3 that describes the next action.
                For "ftj": A dict with keys: "turn" (0 (turn left), 1 (no turn), 2 (turn right)), "forward"
                    (0 (backward), 1(stay), 2 (forward)) and "jump" (0/False (no jump) and 1/True (jump)).

            set_discrete_pos (Optional[int]): An integer to set the current discrete position to before acting.

        Returns:
            tuple: State Space (Space), reward (float), is_terminal (bool), info (usually None).
        """
        # Process possible manual setter instruction.
        #if set_discrete_pos is not None:
        #    assert isinstance(set_discrete_pos, int) and 0 <= set_discrete_pos < self.state_space.flat_dim
        #    self.discrete_pos = set_discrete_pos

        # Forward, turn, jump container action.
        moves = None
        # Up, down, left, right actions.
        if self.action_type == "udlr":
            moves = actions
        else:
            actions = self._translate_action(actions)
            # Turn around (0 (left turn), 1 (no turn), 2 (right turn)).
            if "turn" in actions:
                self.orientations += (actions["turn"] - 1) * 90
                self.orientations %= 360  # re-normalize orientation

            # Forward (0=move back, 1=don't move, 2=move forward).
            if "forward" in actions:
                moves = []
                # Translate into classic grid world action (0=up, 1=right, 2=down, 3=left).
                # We are actually moving in some direction.
                for slot in actions["forward"].shape[0]:
                    forward = actions["forward"][slot]
                    if forward != 1:
                        if self.orientations[slot] == 0 and forward == 2 or self.orientations[slot] == 180 and \
                                forward == 0:
                            moves.append(0)  # up
                        elif self.orientations[slot] == 90 and forward == 2 or self.orientations[slot] == 270 and \
                                forward == 0:
                            moves.append(1)  # right
                        elif self.orientations[slot] == 180 and forward == 2 or self.orientations[slot] == 0 and \
                                forward == 0:
                            moves.append(2)  # down
                        else:
                            moves.append(3)  # left

        if moves is not None:
            moves = np.array(moves)
            # determine the next state based on the transition function
            next_positions, next_positions_probs = self._get_possible_next_positions(self.discrete_pos, moves)
            next_state_indices = np.array(
                [np.random.choice(len(c), p=p) for c, p in zip(next_positions, next_positions_probs)]
            )
            # Update our pos.
            self.discrete_pos = next_positions[np.arange(len(next_state_indices)), next_state_indices]

        # Jump? -> Move two fields forward (over walls/fires/holes w/o any damage).
        if self.action_type == "ftj" and "jump" in actions:
            assert actions["jump"] == 0 or actions["jump"] == 1 or actions["jump"] is True or actions["jump"] is False
            if actions["jump"]:  # 1 or True
                # Translate into "classic" grid world action (0=up, ..., 3=left) and execute that action twice.
                actions = np.cast(self.orientations / 90, np.int32)
                for i in range(2):
                    # Determine the next state based on the transition function.
                    next_positions, next_positions_probs = self._get_possible_next_positions(
                        self.discrete_pos, actions, in_air=(i == 1)
                    )
                    next_state_idx = np.random.choice(len(next_positions), p=next_positions_probs)
                    # Update our pos.
                    self.discrete_pos = next_positions[next_state_idx][0]

        next_x, next_y = self._get_x_y(self.discrete_pos)

        # Determine reward and done flag.
        next_state_types = self.worlds[np.arange(len(next_x)), next_y, next_x]
        for i, next_state_type in enumerate(next_state_types):
            if next_state_type == "H":
                self.terminal[i] = True
                self.reward[i] = -5 if self.reward_function == "sparse" else -10
                self.state[i] = self._reset(i)  # Flow logic (if terminal, state is the new state of a new episode).
            elif next_state_type == "F":
                self.terminal[i] = False
                self.reward[i] = -3 if self.reward_function == "sparse" else -10
                self.state[i] = self._refresh_state(i)
            elif next_state_type in [" ", "S"]:
                self.terminal[i] = False
                self.reward[i] = -0.1
                self.state[i] = self._refresh_state(i)
            elif next_state_type == "G":
                self.terminal[i] = True
                self.reward[i] = 1 if self.reward_function == "sparse" else 50
                self.state[i] = self._reset(i)  # Flow logic (if terminal, state is the new state of a new episode).
            else:
                raise NotImplementedError

    def render(self, num_actors=None, mode="human"):
        actor_slot = 0
        if mode == "human" and pygame is not None:
            self.render_human(actor_slot=actor_slot)
        else:
            print(self.render_txt(actor_slot=actor_slot))

    def render_human(self, actor_slot=0):
        # Set pygame's display, if not already done.
        # TODO: Fix for more than 1 actors.
        if self.pygame_display_set is False:
            pygame.display.set_mode((self.n_col * self.pygame_field_size, self.n_row * self.pygame_field_size))
            self.pygame_display_set = True
        surface = self.pygame_basic_surface.copy()
        surface.blit(self.pygame_agent, (self.x[actor_slot] * self.pygame_field_size + 1, self.y[actor_slot] * self.pygame_field_size + 1))
        pygame.display.get_surface().blit(surface, (0, 0))
        pygame.display.flip()
        pygame.event.get([])

    def render_txt(self, actor_slot=0):
        actor = "X"
        if self.action_type == "ftj":
            actor = "^" if self.orientations[actor_slot] == 0 else ">" if self.orientations[actor_slot] == 90 else \
                "v" if self.orientations[actor_slot] == 180 else "<"

        # paints itself
        txt = ""
        for row in range(len(self.worlds[actor_slot])):
            for col, val in enumerate(self.worlds[actor_slot][row]):
                if self.x == col and self.y == row:
                    txt += actor
                else:
                    txt += val
            txt += "\n"
        txt += "\n"
        return txt

    def __str__(self):
        return "GridWorld({})".format(self.description)

    def _refresh_state(self, actor_slot):
        # Discrete state.
        if self.state_representation == "discrete":
            # TODO: If ftj-actions, maybe multiply discrete states with orientation (will lead to x4 state space size).
            return np.array(self.discrete_pos[actor_slot], dtype=np.int32)
        # xy position.
        elif self.state_representation == "xy":
            return np.array([self.x[actor_slot], self.y[actor_slot]], dtype=np.int32)
        # xy + orientation (only if `self.action_type` supports turns).
        elif self.state_representation == "xy+orientation":
            orient = [0, 1] if self.orientations[actor_slot] == 0 else [1, 0] if self.orientations[actor_slot] == 90 \
                else [0, -1] if self.orientations[actor_slot] == 180 else [-1, 0]
            return np.array([self.x[actor_slot], self.y[actor_slot]] + orient, dtype=np.int32)
        # Camera.
        else:
            return self._update_cam_pixels(actor_slot=actor_slot)

    def _get_possible_next_positions(self, discrete_pos, actions, in_air=False):
        """
        Given discrete positions per-Actor and actions, returns a list of possible next states and
        their probabilities. Only next states with non-zero probabilities will be returned.
        For now: Implemented as a deterministic MDP.

        Args:
            discrete_pos (int): The discrete positions to return possible next states for.
            actions (int): The action choices (per-Actor).
            in_air (bool): Whether we are actually in the air (jumping) right now (ignore if we come from "H" or "W").

        Returns:
            Tuple[np.ndarray[int],np.ndarray[float]]: A tuple of ndarrays (s', p(s'\|s,a)). Where s' are the next
                discrete positions (per Actor) and p(s'|s,a) are the probabilities of ending up in that position
                when in state s and taking action a.
        """
        x, y = self._get_x_y(discrete_pos)
        coords = np.transpose(np.array([x, y]))

        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        next_coords = np.clip(
            coords + increments[actions],
            [0, 0],
            [self.n_col - 1, self.n_row - 1]
        )
        next_pos = self._get_discrete_pos(next_coords[:,0], next_coords[:,1])
        pos_type = self.worlds[:, y, x][:, 0]
        next_pos_type = self.worlds[np.arange(len(self.actors)), next_coords[:, 1], next_coords[:, 0]]
        # TODO: Allow stochasticity in this env. Right now, all probs are 1.0.
        return np.expand_dims(np.where(
            np.logical_or(next_pos_type == "W", (in_air is False and np.logical_or(pos_type == "H", pos_type == "G"))),
            # Next field is a wall or we are already terminal. Stay where we are.
            discrete_pos,
            # Move to next field.
            next_pos
        ), axis=-1), np.ones(shape=discrete_pos.shape + (1,), dtype=np.float)

    def _update_cam_pixels(self, actor_slot):
        self.cam_pixels[actor_slot, :, :, :] = 0  # reset everything

        # 1st channel -> Walls (127) and goal (255).
        # 2nd channel -> Dangers (fire=127, holes=255)
        # 3rd channel -> Actor position (255).
        for row in range(self.n_row):
            for col in range(self.n_col):
                field = self.worlds[actor_slot][row, col]
                if field == "F":
                    self.cam_pixels[row, col, 0] = 127
                elif field == "H":
                    self.cam_pixels[row, col, 0] = 255
                elif field == "W":
                    self.cam_pixels[row, col, 1] = 127
                elif field == "G":
                    self.cam_pixels[row, col, 1] = 255  # will this work (goal==2x wall)?
        # Overwrite player's position.
        self.cam_pixels[actor_slot, self.y, self.x, 2] = 255

        return self.cam_pixels

    def _get_dist_to_goal(self):
        return math.sqrt((self.x - self.goal_x) ** 2 + (self.y - self.goal_y) ** 2)

    def _get_discrete_pos(self, x, y):
        """
        Returns a single, discrete int-value.
        Calculated by walking down the rows of the grid first (starting in upper left corner),
        then along the col-axis.

        Args:
            x (int): The x-coordinate.
            y (int): The y-coordinate.

        Returns:
            int: The discrete pos value corresponding to the given x and y.
        """
        return x * self.n_row + y

    def _get_x_y(self, discrete_pos):
        """
        Returns an x/y tuple given a discrete position.

        Args:
            discrete_pos (np.ndarray[int]): The per-actor int-ndarray describing the discrete position in the grid.

        Returns:
            Tuple[np.ndarray[int],np.ndarray[int]]: Per-actor x and ys as numpy arrays.
        """
        return discrete_pos // self.n_row, discrete_pos % self.n_row

    def _translate_action(self, actions):
        """
        Maps a single integer action to dict actions. This allows us to compare how
        container actions perform when instead using a large range on a single discrete action by enumerating
        all combinations.

        Args:
            actions Union(int, dict): Maps single integer to different actions.

        Returns:
            dict: Actions dict.
        """
        # If already dict, do nothing.
        if isinstance(actions, dict):
            return actions
        else:
            # Unpack
            if isinstance(actions, (np.ndarray, list)):
                actions = actions[0]
            # 3 x 3 x 2 = 18 actions
            assert 18 > actions >= 0
            # For "ftj": A dict with keys: "turn" (0 (turn left), 1 (no turn), 2 (turn right)), "forward"
            # (0 (backward), 1(stay), 2 (forward)) and "jump" (0 (no jump) and 1 (jump)).
            converted_actions = {}

            # Mapping:
            # 0 = 0 0 0
            # 1 = 0 0 1
            # 2 = 0 1 0
            # 3 = 0 1 1
            # 4 = 0 2 0
            # 5 = 0 2 1
            # 6 = 1 0 0
            # 7 = 1 0 1
            # 8 = 1 1 0
            # 9 = 1 1 1
            # 10 = 1 2 0
            # 11 = 1 2 1
            # 12 = 2 0 0
            # 13 = 2 0 1
            # 14 = 2 1 0
            # 15 = 2 1 1
            # 16 = 2 2 0
            # 17 = 2 2 1

            # Set turn via range:
            if 6 > actions >= 0:
                converted_actions["turn"] = 0
            elif 12 > actions >= 6:
                converted_actions["turn"] = 1
            elif 18 > actions >= 12:
                converted_actions["turn"] = 2

            if actions in [0, 1, 6, 7, 12, 13]:
                converted_actions["forward"] = 0
            elif actions in [2, 3, 8, 9, 14, 15]:
                converted_actions["forward"] = 1
            elif actions in [4, 5, 10, 11, 16, 17]:
                converted_actions["forward"] = 2

            # Bool or int as "jump".
            if actions % 2 == 0:
                converted_actions["jump"] = 0 if self.action_type == "ftj" else False
            else:
                converted_actions["jump"] = 1 if self.action_type == "ftj" else True
            return converted_actions

    # png Render helper methods.
    def _grid_to_surface(self):
        """
        Renders the grid-world as a png and returns the png as binary image.

        Returns:

        """
        # Create the png surface.
        surface = pygame.Surface((self.n_col * self.pygame_field_size, self.n_row * self.pygame_field_size), flags=pygame.SRCALPHA)
        surface.fill(pygame.Color("#ffffff"))
        for col in range(self.n_col):
            for row in range(self.n_row):
                x = col * self.pygame_field_size
                y = row * self.pygame_field_size
                pygame.draw.rect(
                    surface, pygame.Color("#000000"), [x, y, self.pygame_field_size, self.pygame_field_size], 1
                )
                # Goal: G
                if self.worlds[0][row][col] in ["G", "S"]:
                    special_field = pygame.font.SysFont("Arial", 24, bold=True).render(
                        self.worlds[0][row][col], False, pygame.Color("#000000")
                    )
                    surface.blit(special_field, (x + 7, y + 1))
                # Wall: W (black rect)
                elif self.worlds[0][row][col] in ["W"]:
                    special_field = pygame.Surface((self.pygame_field_size, self.pygame_field_size))
                    special_field.fill((0, 0, 0))
                    surface.blit(special_field, (x, y))
                # Hole: Hole image.
                elif self.worlds[0][row][col] in ["H"]:
                    special_field = pygame.image.load(
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "images/hole.png")
                    )
                    surface.blit(special_field, (x, y))
                # Fire: F (yellow rect)
                elif self.worlds[0][row][col] in ["F"]:
                    special_field = pygame.image.load(
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "images/fire.png")
                    )
                    #special_field = pygame.Surface((field_size, field_size))
                    #special_field.fill((255, 0, 0) if self.world[row][col] == "H" else (255, 255, 0))
                    surface.blit(special_field, (x, y))
        # Return a png.
        return surface

    def _create_states_heatmap(self, states):
        """
        Generates a heatmap from a list of states.
        """
        state_counts = np.bincount(states)
        alpha = int(255 / np.max(state_counts))
        surface = self.pygame_basic_surface.copy()
        for s in states:
            x, y = self._get_x_y(s)
            #pygame.draw.rect(surface, pygame.Color(0, 255, 0, alpha), [x * field_size, y * field_size, field_size, field_size])
            rect = pygame.Surface((self.pygame_field_size - 2, self.pygame_field_size - 2))
            rect.set_alpha(alpha)
            rect.fill(pygame.Color(0, 255, 0))
            surface.blit(rect, (x * self.pygame_field_size + 1, y * self.pygame_field_size + 1))
        pygame.image.save(surface, "test_states_heatmap.png")

    def _create_states_trajectory(self, states):
        """
        Generates a trajectory from arrows between fields.
        """
        surface = self.pygame_basic_surface.copy()
        for i, s in enumerate(states):
            s_ = states[i + 1] if len(states) > i + 1 else None
            if s_ is not None:
                x, y = self._get_x_y(s)
                x_, y_ = self._get_x_y(s_)
                arrow = pygame.image.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "images/arrow.png"))
                self._add_field_connector(surface, x, x_, y, y_, arrow)
        pygame.image.save(surface, "test_trajectory.png")

    def _create_rewards_trajectory(self, states, rewards):
        """
        Generates a trajectory of received rewards from arrows (green and red) between fields.
        """
        max_abs_r = max(abs(np.array(rewards)))
        surface = self.pygame_basic_surface.copy()
        for i, s in enumerate(states):
            s_ = states[i + 1] if len(states) > i + 1 else None
            if s_ is not None:
                x, y = self._get_x_y(s)
                x_, y_ = self._get_x_y(s_)
                r = rewards[i]
                arrow = pygame.image.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "images/arrow_"+("red" if r < 0 else "green")+".png"))
                arrow_transparent = pygame.Surface((arrow.get_width(), arrow.get_height()), flags=pygame.SRCALPHA)
                arrow_transparent.fill((255, 255, 255, int(255 * ((abs(r) / max_abs_r) / 2 + 0.5))))
                #arrow_transparent.set_alpha(int(255 * abs(r) / max_abs_r))
                #arrow_transparent = pygame.Surface.convert_alpha(arrow_transparent)
                arrow.blit(arrow_transparent, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                self._add_field_connector(surface, x, x_, y, y_, arrow)
        pygame.image.save(surface, "test_rewards_trajectory.png")

    def _add_field_connector(self, surface, x, x_, y, y_, connector_surface):
        # Rotate connector (assumed to be pointing right) according to the direction of the move.
        if x_ == x - 1:  # left
            connector_surface = pygame.transform.rotate(connector_surface, 180.0)
            x = x * self.pygame_field_size - connector_surface.get_width() / 2
            y = y * self.pygame_field_size + (self.pygame_field_size - connector_surface.get_height()) / 2
        elif y_ == y - 1:  # up
            connector_surface = pygame.transform.rotate(connector_surface, 90.0)
            x = x * self.pygame_field_size + (self.pygame_field_size - connector_surface.get_width()) / 2
            y = y * self.pygame_field_size - connector_surface.get_height() / 2
        elif y_ == y + 1:  # down
            connector_surface = pygame.transform.rotate(connector_surface, 270.0)
            x = x * self.pygame_field_size + (self.pygame_field_size - connector_surface.get_width()) / 2
            y = y * self.pygame_field_size + connector_surface.get_height() / 2
        else:  # right
            x = x * self.pygame_field_size + ((self.pygame_field_size * 2) - connector_surface.get_width()) / 2
            y = y * self.pygame_field_size + (self.pygame_field_size - connector_surface.get_height()) / 2
        surface.blit(connector_surface, (x, y))
