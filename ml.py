import pickle
import string

import numpy as np
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

class CRF:
    def __init__(self):
        self.model = pickle.load(open('model/crf.sav', 'rb'))

    def word2features(self, sent, i):
        word = sent[i]

        features = {
            'bias': 1.0,
            'word': word,
            'len(word)': len(word),
            'word[:4]': word[:4],
            'word[:3]': word[:3],
            'word[:2]': word[:2],
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word[-4:]': word[-4:],
            'word.lower()': word.lower(),
            'word.ispunctuation': (word in string.punctuation),
            'word.isdigit()': word.isdigit(),
        }

        if i > 0:
            word1 = sent[i - 1]
            features.update({
                '-1:word': word1,
                '-1:len(word)': len(word1),
                '-1:word.lower()': word1.lower(),
                '-1:word[:3]': word1[:3],
                '-1:word[:2]': word1[:2],
                '-1:word[-3:]': word1[-3:],
                '-1:word[-2:]': word1[-2:],
                '-1:word.isdigit()': word1.isdigit(),
                '-1:word.ispunctuation': (word1 in string.punctuation),
            })
        else:
            features['BOS'] = True

        if i > 1:
            word2 = sent[i - 2]
            features.update({
                '-2:word': word2,
                '-2:len(word)': len(word2),
                '-2:word.lower()': word2.lower(),
                '-2:word[:3]': word2[:3],
                '-2:word[:2]': word2[:2],
                '-2:word[-3:]': word2[-3:],
                '-2:word[-2:]': word2[-2:],
                '-2:word.isdigit()': word2.isdigit(),
                '-2:word.ispunctuation': (word2 in string.punctuation),
            })

        if i < len(sent) - 1:
            word1 = sent[i + 1]
            features.update({
                '+1:word': word1,
                '+1:len(word)': len(word1),
                '+1:word.lower()': word1.lower(),
                '+1:word[:3]': word1[:3],
                '+1:word[:2]': word1[:2],
                '+1:word[-3:]': word1[-3:],
                '+1:word[-2:]': word1[-2:],
                '+1:word.isdigit()': word1.isdigit(),
                '+1:word.ispunctuation': (word1 in string.punctuation),
            })

        else:
            features['EOS'] = True
        if i < len(sent) - 2:
            word2 = sent[i + 2]
            features.update({
                '+2:word': word2,
                '+2:len(word)': len(word2),
                '+2:word.lower()': word2.lower(),
                '+2:word[:3]': word2[:3],
                '+2:word[:2]': word2[:2],
                '+2:word[-3:]': word2[-3:],
                '+2:word[-2:]': word2[-2:],
                '+2:word.isdigit()': word2.isdigit(),
                '+2:word.ispunctuation': (word2 in string.punctuation),
            })

        return features

    def transform(self, sentence):

        # return sentence.replace('[', '').replace(']', '').split()

        X = []

        # words = re.findall(r"\[(.*?)]", sentence)
        # tags = re.findall(r"\((.*?)\)", sentence)

        words = []
        i = j = 0

        try:
            while True:
                i = sentence.index('[')
                j = sentence.index(']')

                X.append(sentence[i + 1:j])
                sentence = sentence[j + 1:]
        except:
            pass

        return X

    def transform_raw(self, sentence):
        x = sentence.replace('.', '').split()
        doubles = []

        for i in range(1, len(x)):
            doubles.append(x[i - 1] + ' ' + x[i])

        return x + doubles

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def predict(self, txt):
        # words = self.transform(txt)
        words = self.transform_raw(txt)

        x = [self.sent2features(words)]

        y = self.model.predict(x)

        print(words, y[0])
        return words, y[0]


class RNN:
    def __init__(self):
        self.model = load_model('model/bi')

        with open('model/word.token', 'rb') as f:
            self.word_tok = pickle.load(f)

        with open('model/tag.token', 'rb') as f:
            self.tag_tok = pickle.load(f)

    def transform(self, sentence):

        # return sentence.replace('[', '').replace(']', '').split()

        X = []

        # words = re.findall(r"\[(.*?)]", sentence)
        # tags = re.findall(r"\((.*?)\)", sentence)

        words = []
        i = j = 0

        try:
            while True:
                i = sentence.index('[')
                j = sentence.index(']')

                X.append(sentence[i + 1:j])
                sentence = sentence[j + 1:]
        except:
            pass

        return X


    def transform_raw(self, sentence):
        x = sentence.replace('.', '').split()
        doubles = []

        for i in range(1, len(x)):
            doubles.append(x[i - 1] + ' ' + x[i])

        return x + doubles

    def predict(self, txt):

        # X = [x for x in self.transform(txt)]
        X = [x for x in self.transform_raw(txt)]
        X_encoded = [[]]
        others = []

        for i, x in enumerate(X):
            t = self.word_tok.texts_to_sequences([[x]])[0]
            if not t:
                others.append(i)  # remember index too
            else:
                X_encoded[0].append(t[0])

        print(X, X_encoded, others)

        # _, X = (list(t) for t in zip(*sorted(zip(X_encoded[0], X))))

        # print(X, X_encoded)

        MAX_SEQ_LENGTH = 100
        X_padded = pad_sequences(X_encoded, maxlen=MAX_SEQ_LENGTH, padding='pre', truncating='post')

        # print(X_padded)
        ypred = self.model.predict(X_padded)[0]

        m = np.argmax(ypred[:, 1:], axis=1)[-len(X_encoded[0]):]
        # m = ypred.argsort()[-len(X_encoded[0]):,-1]
        # print(m)
        tags = [[w + 1] for w in m]
        # print(tags)

        y = self.tag_tok.sequences_to_texts(tags)#[::-1]

        for i in others:
            y.insert(i, 'ot')

        return X, y


if __name__ == '__main__':
    crf = RNN()

    # txt = input('Enter the sentence (keywords in parentheses): ')
    # words, lbls = crf.predict('40 manata albali karti aldim')
    words, lbls = crf.predict('Dunen 40 manata albali karti aldim')
    # words, lbls = crf.predict('Bazar gunu 14:00 bankdan aldim.')
    # words, lbls = crf.predict('Salam, [Universal Avtomatika şirkətinin] bank hesabları sisyemdə görsənmir, onları aktivləşdirməyinizi xahiş edirəm')


    for w, l in zip(words, lbls):
        print(f'{w} -> {l}')
