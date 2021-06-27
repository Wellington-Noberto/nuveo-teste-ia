import unittest
from spam_detection import SpamDetector


class TestSpamDetection(unittest.TestCase):
    sms_detect = SpamDetector("02-SMSSpamDetection/Model/sms_model_v1.pkl")
    spam_test = 'Sunshine Hols. To claim ur med holiday send a stamped self address envelope to Drinks on Us UK, ' \
                'PO Box 113, Bray, Wicklow, Eire. Quiz Starts Saturday! UnsubStop'
    ham_test = 'How. Its a little difficult but its a simple way to enter this place'

    def test_prob_spam(self):
        bool_value = self.sms_detect.prob_spam(self.spam_test)
        self.assertTrue(bool_value > 0.5)

    def test_is_spam(self):
        bool_value = self.sms_detect.is_spam(self.spam_test)
        self.assertTrue(bool_value)

    def test_prob_ham(self):
        bool_value = self.sms_detect.prob_spam(self.ham_test)
        self.assertTrue(bool_value < 0.5)

    def test_is_ham(self):
        bool_value = self.sms_detect.is_spam(self.ham_test)
        self.assertFalse(bool_value)


if __name__ == '__main__':
    unittest.main()
