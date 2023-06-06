import pandas as pd
import warnings
warnings.filterwarnings('ignore')
 
from sklearn.datasets import load_iris, load_boston
from xgboost.sklearn import XGBClassifier

################## df1 ################################
 
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
 
X = df.drop('species', axis=1)
y = df['species']
 
clf = XGBClassifier(
            n_estimators=50, ## 붓스트랩 샘플 개수 또는 base_estimator 개수
            max_depth=5, ## 개별 나무의 최대 깊이
            gamma = 0, ## gamma
            importance_type='gain', ## gain, weight, cover, total_gain, total_cover
            reg_lambda = 1, ## tuning parameter of l2 penalty
            random_state=100
        ).fit(X,y)

## 예측
print(clf.predict(X)[:3]) 
print()
## 성능 평가
print('정확도 : ', clf.score(X,y)) ## 테스트 성능 평가 점수(Accuracy)
print()
## 변수 중요도
for i, feature in enumerate(iris.feature_names):
    print(f'{feature} : {clf.feature_importances_[i]}')

################## df2 ################################