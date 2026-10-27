"""Regression checks for false arithmetic alarms and newly supported syntax."""
import unittest
import numpy as np
from arithmetic_v2 import evidence,value


def check(text):
    return evidence([{'candidates':[{'text':text}]*4}])[0,0]


class ArithmeticTests(unittest.TestCase):
    def test_correct_notation(self):
        for text in (r'20 * 10\% = 2', '20% of 80 = .2*80 = 16',
                     r'3^2 = 9',r'\frac {6}{4} = 1.5',
                     '3(4+2) = 18', 'x = 4\n2*x + 1 = 9',
                     'x + 1 = 9',r'x \times 7 = 994',
                     '0.6/9 = 1/15', '100 - 40 - 10 - 28 = 22'):
            with self.subTest(text=text):
                np.testing.assert_array_equal(check(text)[1:],0)

    def test_inconsistent_steps(self):
        for text in ('5*8 = 45', '0.6/9 = 2/15', '20% of 80 = 18', '3^2 = 6'):
            with self.subTest(text=text):self.assertGreater(check(text)[1],0)
        self.assertEqual(check('x = 4\n2*x + 1 = 10')[3],1)

    def test_unsupported_and_unbounded(self):
        for text in ("__import__('os').system('anything')",'9**99999999','1/0','unknown + 3'):
            with self.subTest(text=text):
                with self.assertRaises((ValueError,ZeroDivisionError)):value(text,{})


if __name__=='__main__':unittest.main()
