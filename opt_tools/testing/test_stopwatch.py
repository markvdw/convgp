import sys
import time
import unittest

sys.path.append('..')
from opt_tools.helpers import Stopwatch


class TestStopwatch(unittest.TestCase):
    def test_basic(self):
        s = Stopwatch()
        s2 = Stopwatch()
        s3 = Stopwatch(elapsed_time=13.0)
        s2.start()
        s3.start()

        time.sleep(0.1)
        self.assertTrue(s.elapsed_time == 0.0)  # Does not start by itself

        s.start()
        time.sleep(0.1)
        s.stop()
        et = s.elapsed_time
        self.assertTrue(et > 0.1)  # Counts the right amount of time

        time.sleep(0.1)
        self.assertTrue(et == s.elapsed_time)  # Does not continue after stopped

        s.start()
        time.sleep(0.1)
        self.assertTrue(s.elapsed_time > 0.2)

        self.assertTrue(s2.elapsed_time > 0.4)  # Other one counts total elapsed time
        self.assertTrue(s3.elapsed_time > 13.0)  # Initialised with elapsed_time works

    def test_pause(self):
        s = Stopwatch()
        s2 = Stopwatch()
        s2.start()

        s.start()
        time.sleep(0.1)
        with s.pause():
            time.sleep(0.1)
        self.assertTrue(0.12 > s.elapsed_time > 0.1)  # Check that it was paused
        time.sleep(0.1)
        self.assertTrue(0.22 > s.elapsed_time > 0.2)  # Check that it's still running

        self.assertTrue(0.32 > s2.elapsed_time > 0.3)
