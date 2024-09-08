import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


# 2a
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Regression:
    def __init__(self, alpha, epoch):
        self.alpha = alpha  # Learning rate
        self.epoch = epoch

    # method used for training
    def train(self, features, labels):
        self.weights = np.zeros(features.shape[1])
        self.bias = 0
        samples = features.shape[0]
        self.loss = []

        # debuging
        print("Starting training...")

        for epoch in range(self.epoch):
            tLoss = 0  # Total loss (E(w_n))
            indices = np.arange(samples)
            np.random.shuffle(indices)  # shuffle data -> Optional

            for i in indices:
                xi = features[i]
                Li = labels[i]  # actual (Li)

                x = np.dot(xi, self.weights) + self.bias  # linear calculation
                lHati = sigmoid(x)  # predicted (lHati)

                # gradients
                dw = (lHati - Li) * xi
                db = lHati - Li

                # weight & bias update
                self.weights -= self.alpha * dw
                self.bias -= self.alpha * db

                # calclate squared error
                loss = np.square(Li - lHati)
                tLoss += loss

            avg_loss = tLoss / samples
            self.loss.append(avg_loss)

    # prediction
    def prediction(self, features):
        p = np.dot(features, self.weights) + self.bias
        return sigmoid(p) >= 0.5


def main():
    # load SpotifyFeatures.csv file
    data = pd.read_csv("/Users/louisasiebel/PycharmProjects/SpotifyFeatures.csv")

    # get number of rows (samples) and columns (fetures)
    sampleCount = data.shape[0]
    featureCount = data.shape[1]

    print(f"Samples: {sampleCount}")
    print(f"Features: {featureCount}")

    # 1b: filter 'Pop' & 'Classical'
    filteredData = data[data['genre'].isin(['Pop', 'Classical'])].copy()  # filter genre
    filteredData['label'] = filteredData['genre'].apply(lambda x: 1 if x == 'Pop' else 0)  # apply label
    popCount = filteredData[filteredData['label'] == 1].shape[0]  # get no. rows with label 1
    classicCount = filteredData[filteredData['label'] == 0].shape[0]  # get no. rows with label 0

    print(f"Pop samples: {popCount}")
    print(f"Classical samples: {classicCount}")

    # 1c:
    matrix = filteredData[['liveness', 'loudness']].to_numpy()  # matrix -> features liveness and loudness
    vector = filteredData['label'].to_numpy()  # vector -> song labels

    # Train-test split (80% train, 20% test) using existing library
    M_train, M_test, V_train, V_test = train_test_split(matrix, vector, test_size=0.2, stratify=vector, random_state=42)
    print(f"Training set size: {M_train.shape[0]}")
    print(f"Test set size: {M_test.shape[0]}")

    # 1d: Graph 1
    plt.scatter(M_train[V_train == 1][:, 0], M_train[V_train == 1][:, 1], color='blue', label='Pop', alpha=0.5)
    plt.scatter(M_train[V_train == 0][:, 0], M_train[V_train == 0][:, 1], color='orange', label='Classical', alpha=0.5)
    plt.xlabel('Liveness')
    plt.ylabel('Loudness')
    plt.title('Liveness vs Loudness')
    plt.legend()
    plt.show()

    alpha = 0.00001
    # 2a: training
    regressionGD = Regression(alpha=alpha, epoch=100)
    regressionGD.train(M_train, V_train)
    print(f"Learning rate: {alpha}")

    # Graph 2: Loss over time -> loss vs Epochs
    plt.plot(range(len(regressionGD.loss)), regressionGD.loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Time')
    plt.show()

    # train Accuracy
    VpTrain = regressionGD.prediction(M_train)
    trainAccuracy = accuracy_score(V_train, VpTrain)
    print(f"Training Accuracy: {trainAccuracy * 100:.2f}%")

    # 2b
    # test accuracy
    Vp = regressionGD.prediction(M_test)
    accuracy = accuracy_score(V_test, Vp)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # weights and bias -> decision boundary plot
    w1, w2 = regressionGD.weights
    b = regressionGD.bias
    x1 = np.linspace(M_train[:, 0].min(), M_train[:, 0].max(), 100)
    x2 = -(w1 / w2) * x1 - (b / w2)

    # Graph 3: decision boundry
    plt.scatter(M_train[V_train == 1][:, 0], M_train[V_train == 1][:, 1], color='blue', label='Pop')
    plt.scatter(M_train[V_train == 0][:, 0], M_train[V_train == 0][:, 1], color='orange', label='Classical')
    plt.plot(x1, x2, color='green', label='Decision Boundary')
    plt.xlabel('Liveness')
    plt.ylabel('Loudness')
    plt.title('Liveness vs Loudness with Decision Boundary')
    plt.legend()
    plt.show()

    # 3a:
    actual_labels = V_test
    predicted_labels = Vp

    conf_matrix = confusion_matrix(actual_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)


if __name__ == '__main__':
    main()

