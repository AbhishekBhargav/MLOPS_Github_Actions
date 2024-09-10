from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

iris = load_iris()
target_labels=iris.target_names
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=42)

sl_min=np.floor(X_train[:,0].min())
sl_max=np.ceil(X_train[:,0].max())

sw_min=np.floor(X_train[:,1].min())
sw_max=np.ceil(X_train[:,1].max())

pl_min=np.floor(X_train[:,2].min())
pl_max=np.ceil(X_train[:,2].max())

pw_min=np.floor(X_train[:,3].min())
pw_max=np.ceil(X_train[:,3].max())

model = RandomForestClassifier()
model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)
print(f"Model accuracy : {accuracy}")

with open('../model.pkl','wb') as f:
    pickle.dump([model,target_labels,[sl_min,sl_max,sw_min,sw_max,pl_min,pl_max,pw_min,pw_max]],f)