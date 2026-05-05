from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

print("Cargando Fashion-MNIST...")
X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=20,
        early_stopping=False,
        random_state=42,
        verbose=True
    ))
])

print("Entrenando...")
pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print(f"Precisión: {accuracy:.4f}")

joblib.dump(pipeline, 'backend/pipeline.pkl')
print("Modelo guardado en backend/pipeline.pkl")