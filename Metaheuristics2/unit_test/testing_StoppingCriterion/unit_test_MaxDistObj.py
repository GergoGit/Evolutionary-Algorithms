import unittest
import numpy as np

# from Differential_Evolution2 import StoppingCriterion
import sys
sys.path.append(r'C:\Users\bonnyaigergo\Documents\GitHub\Evolutionary-Algorithms\Differential_Evolution2')
import StoppingCriterion

class TestMaxDistObj(unittest.TestCase):
    
    def setUp(self):
        self.termination = StoppingCriterion.MaxDistObj()
    
    def test_MaxDistObj_case1(self):
        """
        difference between best and worst fitness is less than tolerance (1E-10)
        -> should meet condition (True)
        but nth_gen is less than from_nth_gen -> should not meet condition (False)

        Returns
        -------
        False
        """
        self.termination.from_nth_gen = 10
        fitness = np.array([0,0,0])
        nth_gen = 2
        self.assertFalse(self.termination.MeetCriterion(fitness, nth_gen))
        
    def test_MaxDistObj_case2(self):
        """
        difference between best and worst fitness is less than tolerance (1E-10)
        -> should meet condition (True)
        and nth_gen (2) is greater than from_nth_gen (1) -> should meet condition (False)

        Returns
        -------
        True
        """
        self.termination.from_nth_gen = 1
        fitness = np.array([0,0,0])
        nth_gen = 2
        self.assertTrue(self.termination.MeetCriterion(fitness, nth_gen))
        
    def test_MaxDistObj_case3(self):
        """
        difference between best and worst fitness is greater than tolerance (1E-10)
        -> should not meet condition (False)
        and nth_gen (2) is greater than from_nth_gen (1) -> should meet condition (True)

        Returns
        -------
        False
        """
        self.termination.from_nth_gen = 1
        fitness = np.array([0, 0, 0.5])
        nth_gen = 2
        self.assertFalse(self.termination.MeetCriterion(fitness, nth_gen))

    def test_MaxDistObj_case4(self):
        """
        difference between best and worst fitness is greater than tolerance (1E-10)
        -> should not meet condition (False)
        and nth_gen (1) is less than from_nth_gen (10) -> should not meet condition (False)

        Returns
        -------
        False
        """
        self.termination.from_nth_gen = 10
        fitness = np.array([0, 0, 0.5])
        nth_gen = 1
        self.assertFalse(self.termination.MeetCriterion(fitness, nth_gen))
        
    def test_MaxDistObj_case5(self):
        """
        Testing outcomes by simulated iterations

        Returns
        -------
        True, True
        """
        self.termination.from_nth_gen = 0
        fitnesses = np.array([[0, 0, 0.5],
                              [0, 0, 0.5],
                              [0, 0, 0]])
        outcome = []
        
        for i in range(3):            
            fitness = fitnesses[i]
            nth_gen = i
            outcome.append(self.termination.MeetCriterion(fitness, nth_gen))
        
        self.assertEqual(self.termination.metric_list.tolist(), [0.5, 0.5, 0])
        self.assertEqual(outcome, [False, False, True])
        
if __name__ == '__main__':
    unittest.main()
    # coverage unittest Unit_tests.py
    # coverage run Unit_tests.py
    # coverage html
