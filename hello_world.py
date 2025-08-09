import numpy as np

import becca.brain as becca_brain
from becca.base_world import World as BaseWorld


class World(BaseWorld):
    """
    To succeed in this world, becca has to guess whether
    the sensor value ("cat" or "dog")
    belongs to the "cat" category or the "dog" category.
    This is a trivial environment, but illustrates
    how to hook up a new world to a becca agent.
    """
    def __init__(self):
        BaseWorld.__init__(self)
        self.name = 'hello_world'
        self.visualize_interval = 1e4

        self.n_sensors = 1
        self.n_actions = 1
        self.sensors = [""]

    def step(self, actions):
        """
        Advance the world environment simulation by one time step.

        Parameters
        ----------
        actions: array of float
            The (one and only) action will be beween 0 and 1.
            If it is above .7, it's a vote for "cat".
            If it's between .4 and .7 it's a vote for "dog".
            If it's below .4, it is not a vote for either.

        Returns
        -------
        sensors: list[str]
            Can be an iterable of any type.
        reward: float
        """
        self.timestep += 1
        if actions[0] >= .7: 
            guess = "cat"
        elif actions[0] >= .4: 
            guess = "dog"
        else:
            guess = "nothing"

        # Is the agent's categorization correct?
        # Check the guess (action) against the previous sensor value.
        if guess == self.sensors[0]:
            reward = 1 
        else:
            reward = -1
        if guess == "nothing":
            reward = 0 

        if self.timestep % self.visualize_interval == 0:
            report = "Timestep " + str(self.timestep) + ". "
            report += "Saw " + self.sensors[0] + ", "
            report += "guessed " + guess + ", "
            report += "got reward of " + str(reward)

            # Give an update
            print(report)

        # Generate a new value for the agent to categorize
        # by flipping a coin.
        if np.random.random_sample() > .5: 
            self.sensors = ["cat"]
        else:
            self.sensors = ["dog"]

        return self.sensors, reward


if __name__ == "__main__":
    becca_brain.run(World())
