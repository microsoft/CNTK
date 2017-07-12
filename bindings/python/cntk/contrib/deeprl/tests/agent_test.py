import unittest

import cntk.contrib.deeprl.tests.spaces as spaces
from cntk.contrib.deeprl.agent.agent import AgentBaseClass


class AgentBaseClassTest(unittest.TestCase):
    """Unit tests for AgentBaseClass."""

    def test_init_unsupported_action_space(self):
        action_space = spaces.Box(0, 1, (1,))
        observation_space = spaces.Discrete(3)
        self.assertRaises(
            ValueError, AgentBaseClass, observation_space, action_space)

    def test_init_unsupported_observation_space(self):
        action_space = spaces.Discrete(2)
        observation_space = spaces.Tuple(
            [spaces.Discrete(3), spaces.Discrete(3)])
        self.assertRaises(
            ValueError, AgentBaseClass, observation_space, action_space)

    def test_init_discrete_observation_space(self):
        action_space = spaces.Discrete(2)
        observation_space = spaces.Discrete(3)
        sut = AgentBaseClass(observation_space, action_space)

        self.assertEqual(sut._num_actions, 2)
        self.assertEqual(sut._num_states, 3)
        self.assertEqual(sut._shape_of_inputs, (3, ))
        self.assertTrue(sut._discrete_observation_space)
        self.assertIsNone(sut._space_discretizer)
        self.assertIsNone(sut._preprocessor)

    def test_init_multibinary_observation_space(self):
        action_space = spaces.Discrete(2)
        observation_space = spaces.MultiBinary(3)
        sut = AgentBaseClass(observation_space, action_space)

        self.assertEqual(sut._num_actions, 2)
        self.assertIsNone(sut._num_states)
        self.assertEqual(sut._shape_of_inputs, (3, ))
        self.assertFalse(sut._discrete_observation_space)
        self.assertIsNone(sut._space_discretizer)
        self.assertIsNone(sut._preprocessor)

    def test_init_box_observation_space(self):
        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = AgentBaseClass(observation_space, action_space)

        self.assertEqual(sut._num_actions, 2)
        self.assertIsNone(sut._num_states)
        self.assertEqual(sut._shape_of_inputs, (1, ))
        self.assertFalse(sut._discrete_observation_space)
        self.assertIsNone(sut._space_discretizer)
        self.assertIsNone(sut._preprocessor)
