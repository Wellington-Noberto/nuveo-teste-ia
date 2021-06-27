import pickle


sms_model = pickle.load(open("02-SMSSpamDetection/Model/sms_model_v1.pkl", "rb"))


class SpamDetector:
    def __init__(self, model_path):
        with open(model_path, "rb") as path:
            self.model = pickle.load(path)

    def prob_spam(self, input_text):
        prob_value = self.model.predict_proba([input_text])[0][1]
        return prob_value

    def is_spam(self, input_text):
        bool_value = self.model.predict([input_text])[0] == 'spam'
        return bool_value

