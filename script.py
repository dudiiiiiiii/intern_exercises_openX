import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Covertype Data Set
file = "covtype.data.gz"
df = pd.read_csv(file, compression='gzip', header=None)

print(df.head())


def heuristic_pred(df):
    y_pred = []
    df1 = df[0]
    if isinstance(df, pd.DataFrame):
        for index, df_row in df.iterrows():
            if df_row[0] < 2000:
                y_pred.append(1)
            elif df_row[0] < 2300:
                y_pred.append(2)
            elif df_row[0] < 2500:
                y_pred.append(3)
            elif df_row[0] < 2700:
                y_pred.append(4)
            elif df_row[0] < 2900:
                y_pred.append(5)
            elif df_row[0] < 3100:
                y_pred.append(6)
            else:
                y_pred.append(7)
    else:
        if df1[0] < 2000:
            y_pred.append(1)
        elif df1[0] < 2300:
            y_pred.append(2)
        elif df1[0] < 2500:
            y_pred.append(3)
        elif df1[0] < 2700:
            y_pred.append(4)
        elif df1[0] < 2900:
            y_pred.append(5)
        elif df1[0] < 3100:
            y_pred.append(6)
        else:
            y_pred.append(7)
    return pd.Series(y_pred)


X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbm = lgb.LGBMClassifier()
gbm.fit(X_train, y_train)

model_tree = DecisionTreeClassifier()
model_tree.fit(X_train, y_train)


# Choosing most important features for later API
# print(model_tree.feature_importances_)
# 0, 5, 9


def train_neural_network(hidden_units, activation, optimizer, epochs):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(units=hidden_units, activation=activation),
        tf.keras.layers.Dense(units=8, activation='softmax')
    ])

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)

    y_pred = model.predict(X_test)
    y_pred = tf.argmax(y_pred, axis=1).numpy()
    acc_nn = accuracy_score(y_test, y_pred)

    return model, acc_nn, history


hidden_units = [16, 32, 64]
activation = ['relu', 'sigmoid']
optimizer = ['adam', 'sgd']
epochs = [10, 20, 30]

best_acc = 0
best_model = None
best_history = None

for units in hidden_units:
    for act in activation:
        for opt in optimizer:
            for ep in epochs:
                model, acc_nn, history = train_neural_network(units, act, opt, ep)
                print(f"Hidden Units: {units}, Activation: {act}, Optimizer: {opt}, Epochs: {ep}")
                print("Accuracy of Neural Network: ", acc_nn)
                print("----------")
                if acc_nn > best_acc:
                    best_acc = acc_nn
                    best_model = model
                    best_history = history

plt.figure(figsize=(10, 5))
plt.plot(best_history.history['accuracy'])
plt.plot(best_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

pred_heuristic = heuristic_pred(X_test)
acc_heuristic = accuracy_score(y_test, pred_heuristic)
print(f'Heuristic accuracy = {acc_heuristic}')

pred_tree = model_tree.predict(X_test)
acc_tree = accuracy_score(y_test, pred_tree)
print(f'Decision tree accuracy = {acc_tree}')

pred = gbm.predict(X_test)
acc_gbm = accuracy_score(y_test, pred)
print(f'LightGBM accuracy = {acc_gbm}')

print(f'Neural Network accuracy = {best_acc}')

from flask import Flask, request, jsonify, render_template, redirect, url_for

app = Flask(__name__)


@app.route('/')
def get_data():
    return redirect(url_for('predict'))


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        model_choice = request.form.get('model_choice')
        feature1 = request.form.get('elevation')
        feature2 = request.form.get('hor_dist_road')
        feature3 = request.form.get('hor_dist_fire')

        tmp = pd.Series([0] * 54)
        tmp.iloc[0] = int(feature1)
        tmp.iloc[5] = int(feature2)
        tmp.iloc[9] = int(feature3)
        tmp = tmp.values.reshape((1, 54))
        print(tmp)

        if model_choice == "heuristic":
            prediction = int(heuristic_pred(tmp).iloc[0])

        elif model_choice == "lightgbm":
            prediction = int(gbm.predict(tmp))

        elif model_choice == "tree":
            prediction = int(model_tree.predict(tmp))

        elif model_choice == 'nn':
            prediction = int(best_model.predict(tmp))

        return jsonify({'prediction': prediction})

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
