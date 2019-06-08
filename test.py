#load dataset
import mimicus.tools.datasets
X, y, names = mimicus.tools.datasets.csv2numpy("train.csv")


#train model
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_estimators=100, n_jobs=-1, oob_score=True)
model.fit(X,y)

#info about model
model.oob_score_
model.feature_importances_

#predict, predict showing votes
model.predict(X)
model.predict_proba(X)

#leaf nodes
model.apply(X)

#serialize/deserialize classifier
import pickle
pickle.dump(model, open("example.model", 'wb+'))
model = pickle.load(open("example.model", 'rb+'))
