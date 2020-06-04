
import os

class YelpTopicClassification:
    def __init__(self):
        self.food_vocab = None
        self.atmosphere_vocab = None
        self.service_vocab = None
        self.price_vocab = None

    def loadData(self):
        path = os.path.join(os.path.dirname(__file__), "..\\data\\foods.txt")
        data = open(path)
        self.food_vocab = data.readlines()[0].split(',')
        self.food_vocab = list(set(self.food_vocab))
        data.close()

        path = os.path.join(os.path.dirname(__file__), "..\\data\\atmosphere.txt")
        data = open(path)
        self.atmosphere_vocab = data.readlines()[0].split(',')
        self.atmosphere_vocab = list(set(self.atmosphere_vocab))
        data.close()

        path = os.path.join(os.path.dirname(__file__), "..\\data\\service.txt")
        data = open(path)
        self.service_vocab = data.readlines()[0].split(',')
        self.service_vocab = list(set(self.service_vocab))
        data.close()

        path = os.path.join(os.path.dirname(__file__), "..\\data\\price.txt")
        data = open(path)
        self.price_vocab = data.readlines()[0].split(',')
        self.price_vocab = list(set(self.price_vocab))
        data.close()