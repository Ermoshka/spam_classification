import numpy as np
import re

class Dataset:
    def __init__(self, X, y):
        self._x = X # сообщения 
        self._y = y # метки ["spam", "ham"]
        self.train = None # кортеж из (X_train, y_train)
        self.val = None # кортеж из (X_val, y_val)
        self.test = None # кортеж из (X_test, y_test)
        self.num2label = {
            1: "spam",
            0: "ham"
        }
        self.label2num = {
            "spam": 1,
            "ham": 0
        }
        self._transform()
        
    def __len__(self):
        return len(self._x)
    
    def _transform(self):
        '''
        Функция очистки сообщения и преобразования меток в числа.
        '''
        # Начало вашего кода
        vf = np.vectorize(lambda x: re.sub(r"\W+", " ", x.lower()))
        self._x = vf(self._x.tolist())
        self._y[self._y == "spam"] = self.label2num["spam"]
        self._y[self._y == "ham"] = self.label2num["ham"]
        # Конец вашего кода
        pass

    def split_dataset(self, val=0.1, test=0.1):
        '''
        Функция, которая разбивает набор данных на наборы train-validation-test.
        '''
        # Начало вашего кода
        np.random.seed(1)
        indices = np.arange(0, len(self._x))
        np.random.shuffle(indices)
        self.val = (
            self._x[:indices[round(val*len(self._x))]],
            self._y[:indices[round(val*len(self._y))]]
        )
        self.test = (
            self._x[indices[round(test*len(self._x))]:indices[round(0.2*len(self._x))]],
            self._y[indices[round(test*len(self._y))]:indices[round(0.2*len(self._y))]]
        )
        self.train = (
            self._x[indices[round(0.2*len(self._x))]:],
            self._y[indices[round(0.2*len(self._y))]:]
        )
        return self
        # Конец вашего кода
