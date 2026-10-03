"""Small evaluator tests; no metaheuristic experiment or solver is run."""

from __future__ import annotations

import copy
import itertools
import math
import random
import unittest

from experiment_common import benchmark
import lemma2_metaheuristics as meta


class DecoderTests(unittest.TestCase):
    def setUp(self):
        raw = benchmark.build_raw_instance(benchmark.BenchmarkCase("test", 4, 2, 0.2), 21)
        self.data = meta.prepare_data(raw)

    def recurrence(self, seq, agv):
        value = self.data["t0"]
        for job in seq:
            value = self.data["q"][job][agv] * value + self.data["alpha"][job][agv]
        return value

    def test_decoder_minimizes_fixed_assignment(self):
        vector = [0.5, 0.5, 0.5, 1.5, 4.0, 3.0, 2.0, 1.0]
        original = vector[:]
        solution = meta.decode(vector, self.data)
        self.assertEqual(vector, original)
        self.assertEqual(solution.vector, original)
        self.assertEqual(solution.assignment, [0, 0, 0, 1])
        for agv, seq in enumerate(solution.sequences):
            indices = [self.data["rho"][job][agv] for job in seq]
            self.assertEqual(indices, sorted(indices))
            minimum = min(self.recurrence(order, agv) for order in itertools.permutations(seq))
            self.assertAlmostEqual(solution.completion[agv], minimum)
        self.assertAlmostEqual(solution.cmax, max(solution.completion))

    def test_partial_evaluation_sorts_and_does_not_change_input(self):
        seq = [2, 0, 1]
        optimum = min(self.recurrence(order, 0) for order in itertools.permutations(seq))
        self.assertAlmostEqual(meta.engine.agv_sequence_completion(seq, 0, self.data), optimum)
        self.assertEqual(seq, [2, 0, 1])
        self.assertAlmostEqual(meta.engine.partial_cmax([seq, [3]], self.data), max(optimum, self.recurrence([3], 1)))

    def test_priorities_only_break_ties(self):
        self.data["q"][0][0] = self.data["q"][1][0] = 2.0
        self.data["alpha"][0][0] = self.data["alpha"][1][0] = 3.0
        self.data["rho"][0][0] = self.data["rho"][1][0] = 3.0
        vector = [0.5, 0.5, 1.5, 1.5, 10.0, -10.0, 0.0, 0.0]
        self.assertEqual(meta.decode(vector, self.data).sequences[0], [1, 0])
        self.data["rho"][0][0] = 2.0
        self.data["alpha"][0][0] = 2.0
        self.assertEqual(meta.decode(vector, self.data).sequences[0], [0, 1])

    def test_non_deteriorating_positive_task_is_last(self):
        self.data["q"][0][0] = 1.0
        self.data["alpha"][0][0] = 2.0
        self.data["rho"][0][0] = math.inf
        solution = meta.decode([0.5, 0.5, 1.5, 1.5, -100.0, 100.0, 0.0, 0.0], self.data)
        self.assertEqual(solution.sequences[0], [1, 0])

    def test_capacity_repair_and_empty_agv(self):
        self.data["Q"][0] = 0.1
        solution = meta.decode([0.5] * 4 + [0.0] * 4, self.data)
        self.assertEqual(solution.assignment, [1] * 4)
        self.assertEqual(solution.completion[0], self.data["t0"])
        with self.assertRaises(ValueError):
            meta.decode([0.0], self.data)

    def test_all_evaluation_paths_use_installed_hooks(self):
        for function in (meta.engine.solve_sa, meta.engine.solve_ts, meta.engine.solve_ga_sa, meta.engine.solve_pso_sa):
            self.assertIs(function.__globals__["decode"], meta.decode)
            self.assertIs(function.__globals__["agv_sequence_completion"], meta.agv_sequence_completion)
        evaluated = meta.engine.evaluate_sequences([[2, 0, 1], [3]], self.data)
        self.assertAlmostEqual(evaluated.completion[0], meta.agv_sequence_completion([0, 1, 2], 0, self.data))

    def test_initial_solution_matches_vns_he(self):
        reference = benchmark.load_module(benchmark.PROJECT_ROOT / "VNS-HE.py", "_lemma2_test_reference")
        data = reference.prepare_data(copy.deepcopy(self.data))
        expected = reference.build_initial_solution(data, random.Random(10))
        actual = meta.build_initial_solution(self.data, random.Random(99))
        self.assertEqual(actual.assignment, expected.assignment)
        self.assertEqual(actual.sequences, expected.sequences)
        self.assertAlmostEqual(actual.cmax, expected.cmax)


if __name__ == "__main__":
    unittest.main()
