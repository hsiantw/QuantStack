import unittest
from unittest.mock import Mock, patch

import idle_collector as idle


class IdleCollectorTests(unittest.TestCase):
    def test_threshold_and_fail_closed(self):
        for seconds, expected in [(0, False), (899, False), (900, False), (901, True)]:
            with patch.object(idle, 'idle_seconds', return_value=seconds):
                self.assertEqual(idle.allowed(), expected)
        with patch.object(idle, 'idle_seconds', side_effect=OSError):
            self.assertFalse(idle.allowed())

    def test_no_process_started_while_active(self):
        with patch.object(idle, 'allowed', return_value=False), patch.object(idle.subprocess, 'Popen') as start:
            self.assertEqual(idle.run(['collector']), 0)
            start.assert_not_called()

    def test_returning_input_stops_process_tree(self):
        child = Mock()
        child.poll.side_effect = [None, 0]
        with patch.object(idle, 'allowed', side_effect=[True, False]), \
             patch.object(idle.subprocess, 'Popen', return_value=child), \
             patch.object(idle, 'stop_tree') as stop:
            self.assertEqual(idle.run(['collector']), 0)
            stop.assert_called_once_with(child)

    def test_completed_child_exit_code(self):
        child = Mock(returncode=7)
        child.poll.return_value = 7
        with patch.object(idle, 'allowed', return_value=True), \
             patch.object(idle.subprocess, 'Popen', return_value=child):
            self.assertEqual(idle.run(['collector']), 7)
