import unittest
import numpy as np

# from Differential_Evolution2 import StoppingCriterion
import sys
sys.path.append(r'C:\Users\bonnyaigergo\Documents\GitHub\Evolutionary-Algorithms\Differential_Evolution2')
import StoppingCriterion

class TestImpBestObj(unittest.TestCase):
    
    def setUp(self):
        self.termination = StoppingCriterion.ImpBestObj()
        
    def test_ImpBestObj_case1(self):
        """
        Testing outcomes by simulated iterations

        Returns
        -------
        True, True
        """
        self.termination.from_nth_gen = 0
        self.termination.patience = 3
        fitnesses = np.array([[0, 2, 1],
                              [0, 1, 1],
                              [0, 0, 1],
                              [0, 0, 1]])
        best_idx = 0
        outcome = []
        
        for i in range(len(fitnesses)):            
            fitness = fitnesses[i]
            nth_gen = i
            outcome.append(self.termination.MeetCriterion(fitness, best_idx, nth_gen))
        
        self.assertEqual(self.termination.metric_list.tolist(), [0, 0, 0, 0])
        self.assertEqual(outcome, [False, False, True, True])
        
if __name__ == '__main__':
    unittest.main()
    # Pycharm
    # import coverage
    # coverage run Differential_Evolution2\unit_test\testing_StoppingCriterion\unit_test_MaxDistObj.py
    # coverage run -m unittest Differential_Evolution2\unit_test\testing_StoppingCriterion\unit_test_MaxDistObj.py
    # coverage report
    # coverage report -m
    # coverage html
    # index.html file
    
    # pip install pytest-cov
