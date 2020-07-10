import numpy as np
import pandas as pd
import re
from dataset import Dataset

class Model:
    def __init__(self, alpha=1):
        self.vocab = {} # словарь, содержащий все уникальные слова из набора train
        self.spam = {} # словарь, содержащий частоту слов в спам-сообщениях из набора данных train.
        self.ham = {} # словарь, содержащий частоту слов в не спам-сообщениях из набора данных train.
        self.alpha = alpha # сглаживание
        self.num2label = {1: "spam", 0: "ham"}
        self.label2num = {"spam": 1, "ham": 0}
        self.Nvoc = None # общее количество уникальных слов в наборе данных train
        self.Nspam = None # общее количество уникальных слов в спам-сообщениях в наборе данных train
        self.Nham = None # общее количество уникальных слов в не спам-сообщениях в наборе данных train
        self._train_X, self._train_y = None, None
        self._val_X, self._val_y = None, None
        self._test_X, self._test_y = None, None

    def _calc_probs(self, sam_vocab, vocab, message, alpha=1):
        msg, Ns, Nv, Pw = re.sub(r"\W+", " ", message.lower()), 0, len(vocab), 1
        Ns = sum(list(sam_vocab.values()))
        for w in msg.split(" "):
            Nsw = 0
            if w in sam_vocab:
                Nsw = sam_vocab[w]
            Pw *= (Nsw + alpha) / (Ns + alpha*Nv)
        return Pw
    
    def _calc_msg(self, msg, condition=None):
        data = pd.DataFrame({"status": msg[1].tolist(), "message": msg[0].tolist()})
        m = data['message'].tolist()
        if condition is not None:
            m = data[data['status'] == condition]['message'].tolist()
        md = {}
        for s in m:
            for w in s.split(" "):
                if w not in md:
                    md[w] = 0
                md[w] += 1
        return md
    
    def fit(self, dataset):
        '''
        dataset - объект класса Dataset
        Функция использует входной аргумент "dataset", 
        чтобы заполнить все атрибуты данного класса.
        '''
        # Начало вашего кода
        self._train_X, self._train_y = dataset.train[0], dataset.train[1]
        self._val_X, self._val_y = dataset.val[0], dataset.val[1]
        self._test_X, self._test_y = dataset.test[0], dataset.test[1]
        self.spam = self._calc_msg((self._train_X, self._train_y), 1)
        self.ham = self._calc_msg((self._train_X, self._train_y), 0)
        self.vocab = self._calc_msg((self._train_X, self._train_y))
        self.Nvoc = len(self.vocab)
        self.Nspam = len(self.spam)
        self.Nham = len(self.ham)

        # Конец вашего кода
        return self
    
    def inference(self, message):
        '''
        Функция принимает одно сообщение и, используя наивный байесовский алгоритм, определяет его как спам / не спам.
        '''
        # Начало вашего кода
        alls = sum(list(self.spam.values())) + sum(list(self.ham.values()))
        spams = sum(list(self.spam.values()))
        hams = sum(list(self.ham.values()))
        pspam = (spams/alls) * self._calc_probs(self.spam, self.vocab, message)
        pham = (hams/alls) * self._calc_probs(self.ham, self.vocab, message)
        
        # Конец вашего кода
        if pspam > pham:
            return "spam"
        return "ham"
    
    def validation(self):
        '''
        Функция предсказывает метки сообщений из набора данных validation,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().
        '''
        # Начало вашего кода
        spams = [i for i in self._val_X if self.inference(i) == "spam"]
        hams = [i for i in self._val_X if self.inference(i) == "ham"]
        val_acc = {
            "spams": f"{len(spams)*100//len(self._val_y[self._val_y == 1 ])}%", 
            "hams": f"{len(hams)*100//len(self._val_y[self._val_y == 0 ])}%"
        }
        # Конец вашего кода
        return val_acc 

    def test(self):
        '''
        Функция предсказывает метки сообщений из набора данных test,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().
        '''
        # Начало вашего кода
        spams = [i for i in self._val_X if self.inference(i) == "spam"]
        hams = [i for i in self._val_X if self.inference(i) == "ham"]
        val_acc = {
            "spams": f"{len(spams)*100//len(self._val_y[self._val_y == 1 ])}%", 
            "hams": f"{len(hams)*100//len(self._val_y[self._val_y == 0 ])}%"
        }
        # Конец вашего кода
        return val_acc 


