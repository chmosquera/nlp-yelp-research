

class YelpTopicClassification:
    def __init__(self):
        self.food_vocab = None
        self.atmosphere_vocab = None
        self.service_vocab = None
        self.price_vocab = None

    def loadData(self):
        data = open("data/foods.txt")
        self.food_vocab = data.readlines()[0].split(',')
        self.food_vocab = list(set(self.food_vocab))
        data.close()

        data = open("data/atmosphere.txt")
        self.atmosphere_vocab = data.readlines()[0].split(',')
        self.atmosphere_vocab = list(set(self.atmosphere_vocab))
        data.close()

        data = open("data/service.txt")
        self.service_vocab = data.readlines()[0].split(',')
        self.service_vocab = list(set(self.service_vocab))
        data.close()

        data = open("data/price.txt")
        self.price_vocab = data.readlines()[0].split(',')
        self.price_vocab = list(set(self.price_vocab))
        data.close()