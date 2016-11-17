# Python-MLP
Multi Layer Perceptron written in Python3 with scikit-lean

## How to install 
```
pip install git+https://github.com/masaponto/python-mlp
```

## Require
- scikit-learn: 0.17.1 and up
- numpy: 1.10.4 and up


## Usage
```python
from mlp import MLP
from sklearn.datasets import fetch_mldata
from sklearn import preprocessing
from sklearn import cross_validation

db_name = 'iris'

data_set = fetch_mldata(db_name)
data_set.data = preprocessing.scale(data_set.data)

mlp = MLP(hid_nums=[5], epochs=2000, batch_size=30)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    data_set.data, data_set.target, test_size=0.4)

mlp.fit(X_train, y_train)

print("Accuracy %0.3f " % mlp.score(X_test, y_test))
```


## Reference
[機械学習プロフェッショナルシリーズ 深層学習](http://www.kspub.co.jp/book/detail/1529021.html)
