import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 1A
df_pd = pd.read_csv("SpotifyFeatures.csv")

print(df_pd.shape[0])
print(df_pd.shape[1])

# 1B
df_pd = df_pd[df_pd["genre"].isin(["Pop", "Classical"])]

df_pd["label"] = df_pd["genre"].apply(lambda x: 1 if x == "Pop" else 0)

df_pd_filtered = df_pd[["label", "liveness", "loudness"]]

# 1C
data = df_pd_filtered[["liveness", "loudness"]].values
labels = df_pd_filtered["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels
)

# 1D
for label in np.unique(y_train):
    plt.scatter(
        X_train[y_train == label, 0],
        X_train[y_train == label, 1],
        label=("Pop" if label == 1 else "Classical"),
    )
plt.xlabel("liveness")
plt.ylabel("loudness")
plt.legend()
plt.show()


# 2A
class LogisticDiscriminationClassifier:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.errors = []

    def compute_gradient(self, features, y, y_hat):
        return features * (y_hat - y)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, features):
        dot = np.dot(features, self.weights)
        return self.sigmoid(dot)

    def predict_class(self, vals):
        y_hat = self.predict(vals)
        y_class = [1 if i > 0.5 else 0 for i in y_hat]
        return np.array(y_class)

    def loss(self, y, y_hat):
        return -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

    def fit(self, data, label):
        # Init
        samples_count = data.shape[0]
        features_count = data.shape[1]

        self.weights = np.zeros(features_count)
        self.bias = 0
        # run epochs
        # using the pseudo code from the slides with the implemented formulas and with added error tracking
        for i in range(self.epochs):
            curr_epoch_loss = 0

            # random feed
            # https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html
            shuffled_i = np.arange(samples_count)
            np.random.shuffle(shuffled_i)
            data = data[shuffled_i]
            label = label[shuffled_i]

            for j in range(samples_count):
                features = data[j]
                y = label[j]

                y_hat = self.predict(features)
                gradient = self.compute_gradient(features, y, y_hat)

                self.weights -= self.learning_rate * gradient
                self.bias -= self.learning_rate * (y_hat - y)

                curr_epoch_loss += self.loss(y, y_hat)

            avg_epoch_loss = curr_epoch_loss / samples_count
            self.errors.append(avg_epoch_loss)

        return self.errors


# Init model
model = LogisticDiscriminationClassifier(learning_rate=0.001, epochs=50)

# Train model
errors = model.fit(X_train, y_train)

# See error
plt.plot(range(model.epochs), errors)
plt.xlabel("epochs")
plt.ylabel("error")
plt.legend()
plt.show()

# predict for train
train_predict = model.predict_class(X_train)

# see train acc
print(f"train acc {accuracy_score(y_train, train_predict)}")

# 2B
test_predict = model.predict_class(X_test)

print(f"test acc {accuracy_score(y_test, test_predict)}")

# 3A
print(confusion_matrix(y_test, test_predict))
