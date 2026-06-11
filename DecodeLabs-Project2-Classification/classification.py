import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=" * 50)
    print("  DATA CLASSIFICATION USING AI - Project 2")
    print("=" * 50)

    # Step 1: Load and understand dataset
    print("\n[1] Loading Iris Dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(f"    Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"    Classes: {list(iris.target_names)}")

    # Step 2: Split data into training and testing sets
    print("\n[2] Splitting data (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling for better accuracy
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 3: Apply a simple classification algorithm (KNN)
    print("\n[3] Training K-Nearest Neighbors (KNN) model...")
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_scaled, y_train)
    print("    Model trained successfully!")

    # Step 4: Evaluate model
    print("\n[4] Evaluating Model on Test Data...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"    Accuracy: {accuracy * 100:.2f}%")
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Bonus: Interactive Prediction
    print("\n" + "=" * 50)
    print("  BONUS: Predict your own Iris flower!")
    print("=" * 50)
    try:
        print("Enter measurements in cm (e.g., 5.1, 3.5, 1.4, 0.2):")
        sl = float(input("  Sepal Length: "))
        sw = float(input("  Sepal Width: "))
        pl = float(input("  Petal Length: "))
        pw = float(input("  Petal Width: "))

        sample = np.array([[sl, sw, pl, pw]])
        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)[0]

        print(f"\n  --> Predicted Species: {iris.target_names[prediction].upper()} <--")
    except ValueError:
        print("  Invalid input. Skipping prediction.")

    print("\nProject 2 completed!")

if __name__ == "__main__":
    main()