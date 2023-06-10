import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_probabilities = None
        self.feature_probabilities = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_probabilities = self.calculate_class_probabilities(y)
        self.feature_probabilities = self.calculate_feature_probabilities(X, y)

    def calculate_class_probabilities(self, y):
        class_probabilities = {}
        total_samples = len(y)

        for class_value in self.classes:
            class_samples = np.sum(y == class_value)
            class_probabilities[class_value] = class_samples / total_samples

        return class_probabilities

    def calculate_feature_probabilities(self, X, y):
        feature_probabilities = {}

        for feature in range(X.shape[1]):
            feature_probabilities[feature] = {}
            unique_feature_values = np.unique(X[:, feature])

            for class_value in self.classes:
                feature_probabilities[feature][class_value] = {}

                for value in unique_feature_values:
                    samples_with_class = X[y == class_value]
                    feature_value_count = np.sum(samples_with_class[:, feature] == value)
                    total_samples_with_class = len(samples_with_class)
                    probability = (feature_value_count + 1) / (total_samples_with_class + len(unique_feature_values))
                    feature_probabilities[feature][class_value][value] = probability

        return feature_probabilities

    def predict(self, X):
        predictions = []

        for sample in X:
            probabilities = {}

            for class_value in self.classes:
                class_probability = self.class_probabilities[class_value]
                feature_probabilities = self.feature_probabilities

                for feature, value in enumerate(sample):
                    if value in feature_probabilities[feature][class_value]:
                        feature_probability = feature_probabilities[feature][class_value][value]
                    else:
                        feature_probability = 1 / len(feature_probabilities[feature][class_value])

                    class_probability *= feature_probability

                probabilities[class_value] = class_probability

            predicted_class = max(probabilities, key=probabilities.get)
            predictions.append(predicted_class)

        return predictions


data = pd.read_excel('C:\\Users\\gg_ba\\Downloads\\adult.xlsx')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
naive_bayes = NaiveBayes()
naive_bayes.fit(X, y)

predictions = naive_bayes.predict(X)

color_map = {class_value: idx for idx, class_value in enumerate(np.unique(y))}
colors = [color_map[class_value] for class_value in y]

plt.scatter(range(len(predictions)), predictions, c=colors, cmap='viridis')
plt.xlabel('Amostras')
plt.ylabel('Previsões')
plt.title('Gráfico')
plt.show()
