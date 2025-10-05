from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('keplar.csv', comment='#')


features = [
	'koi_period',
	'koi_duration',
	'koi_depth',
	'koi_prad',
	'koi_model_snr'
]
target = 'koi_disposition'

df = df.dropna(subset=features + [target])

df['is_false_positive'] = (df[target] == 'FALSE POSITIVE').astype(int)

X = df[features]
y = df['is_false_positive']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



param_grid = {
	'n_estimators': [100, 200],
	'max_depth': [None, 10, 20],
	'min_samples_split': [2, 5],
	'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print('Best parameters:', grid_search.best_params_)
model = grid_search.best_estimator_


y_pred = model.predict(X_test)
joblib.dump(model, 'rf_model.joblib')
print('Model saved to rf_model.joblib')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=['Not False Positive', 'False Positive']))

