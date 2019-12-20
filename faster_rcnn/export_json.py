from config import Config
from model import build_model

JSON_RPN = './model/rpn.json'
JSON_CLASS = './model/classifier.json'


conf = Config()
modelparts = build_model(conf)
rpn, classifier = modelparts
rpn.load_weights(conf.weights_path, by_name=True)
classifier.load_weights(conf.weights_path, by_name=True)
rpn.compile(optimizer='sgd', loss='mse')
classifier.compile(optimizer='sgd', loss='mse')

rpn_json = rpn.to_json()
class_json = classifier.to_json()

with open(JSON_RPN, 'w') as file:
    file.write(rpn_json + '\n')

with open(JSON_CLASS, 'w') as file:
    file.write(class_json + '\n')
