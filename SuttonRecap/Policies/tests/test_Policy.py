from unittest import TestCase

from Policies import GreedyPolicy, EpsilonGreedyPolicy, TabularValueFunction


class TestTabularPolicy(TestCase):
    example_policy = {"s1": {"a1": 1, "a2": 2}, "s2": {"a1": 3, "a2": 2}, "s3": {"a1": 1, "a2": 1}}
    q = TabularValueFunction(lambda x: None, example_policy)

    def test_get(self):
        p = GreedyPolicy(self.q)
        self.assertEqual({"a1": 0, "a2": 1}, p.get_prob("s1"))
        self.assertEqual({"a1": 1, "a2": 0}, p.get_prob("s2"))
        self.assertEqual({"a1": 0.5, "a2": 0.5}, p.get_prob("s3"))

    def test_get_epsilon_greedy(self):
        pb = GreedyPolicy(self.q)
        p = EpsilonGreedyPolicy(pb)
        self.assertEqual({"a1": 0.1, "a2": 0.9}, p.get_prob("s1", 0.2))
        self.assertEqual({"a1": 0.5, "a2": 0.5}, p.get_prob("s1", 1))
        self.assertEqual({"a1": 0.5, "a2": 0.5}, p.get_prob("s3", 0.2))
