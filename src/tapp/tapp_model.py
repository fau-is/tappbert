import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import os
from tensorflow.keras import metrics
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
from .prediction_model import PredictionModel
from sklearn.metrics import f1_score, precision_score, recall_score
import math


class TappModel(PredictionModel):

    def __init__(self, log_encoder=None, num_shared_layer=1, num_specialized_layer=1, neurons_per_layer=100, dropout=0.2,
                 learning_rate=0.002):
        self.log_encoder = log_encoder
        self.num_shared_layer = num_shared_layer
        self.num_specialized_layer = num_specialized_layer
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.activities = []
        self.timesteps = 0
        self.feature_dim = 0
        self.model = None
        super().__init__()

    def _build_model(self):
        # Input layer
        inputs = tf.keras.Input(shape=(self.timesteps, self.feature_dim), name="inputs")

        # Shared layer
        previous_layer = inputs
        for layer in range(self.num_shared_layer):
            # Do not return sequences in the last LSTM layer
            return_sequences = False if self.num_specialized_layer == 0 and layer == self.num_shared_layer - 1 else True
            shared_lstm_layer = layers.LSTM(self.neurons_per_layer, implementation=2, kernel_initializer="glorot_uniform",
                                            return_sequences=return_sequences, dropout=self.dropout)(previous_layer)
            shared_lstm_normalization_layer = layers.LayerNormalization()(shared_lstm_layer)
            previous_layer = shared_lstm_normalization_layer

        previous_layer_next_activity = previous_layer
        previous_layer_next_timestamp = previous_layer
        for layer in range(self.num_specialized_layer):
            # Do not return sequences in the last LSTM layer
            return_sequences = False if layer == self.num_specialized_layer - 1 else True
            next_activity_lstm_layer = layers.LSTM(self.neurons_per_layer, implementation=2,
                                                   kernel_initializer="glorot_uniform", return_sequences=return_sequences,
                                                   dropout=self.dropout)(previous_layer_next_activity)
            next_timestamp_lstm_layer = layers.LSTM(self.neurons_per_layer, implementation=2,
                                                    kernel_initializer="glorot_uniform", return_sequences=return_sequences,
                                                    dropout=self.dropout)(previous_layer_next_timestamp)
            next_activity_normalization_layer = layers.LayerNormalization()(next_activity_lstm_layer)
            next_timestamp_normalization_layer = layers.LayerNormalization()(next_timestamp_lstm_layer)
            previous_layer_next_activity = next_activity_normalization_layer
            previous_layer_next_timestamp = next_timestamp_normalization_layer

        # Output layer
        next_activity_output = layers.Dense(len(self.activities) + 1, activation="softmax",
                                            kernel_initializer="glorot_uniform",
                                            name="next_activity_output")(previous_layer_next_activity)
        next_timestamp_ouput = layers.Dense(1, kernel_initializer="glorot_uniform",
                                            name="next_timestamp_output")(previous_layer_next_timestamp)

        # Build and configure model
        model = tf.keras.Model(inputs=[inputs], outputs=[next_activity_output, next_timestamp_ouput])
        optimizer_param = tf.keras.optimizers.Nadam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999,
                                                    epsilon=1e-07, name="Nadam")
        metric_param = {"next_activity_output": metrics.CategoricalAccuracy(),
                        "next_timestamp_output": metrics.MeanAbsoluteError()}
        loss_param = {"next_activity_output": "categorical_crossentropy", "next_timestamp_output": "mae"}
        model.compile(loss=loss_param, metrics=metric_param, optimizer=optimizer_param)
        self.model = model

    def fit(self, log, data_attributes=None, text_attribute=None, epochs=100, validation_split=0.2, patience_rl=5,
            patience_es=10, batch_size=None):
        # Encode training data
        # self.activities = _get_event_labels(log, "concept:name")
        # self.log_encoder.fit(log, activities=self.activities, data_attributes=data_attributes, text_attribute=text_attribute)
        x, y_next_act, y_next_time = self.log_encoder.transform(log, for_training=True)

        # Build model
        self.timesteps = x.shape[1]
        self.feature_dim = x.shape[2]
        self._build_model()

        # Reduce learning rate if metrics do not improve anymore
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=patience_rl, min_lr=0.0001)
        # Stop early if metrics do not improve for longer time
        early_stopping = EarlyStopping(monitor="val_loss", patience=patience_es)
        # Fit the model to data
        return self.model.fit(x, {"next_activity_output": y_next_act, "next_timestamp_output": y_next_time}, epochs=epochs,
                              batch_size=batch_size, validation_split=validation_split,
                              callbacks=[reduce_lr, early_stopping])

    def predict_next_activity(self, log):
        x = self.log_encoder.transform(log, for_training=False)
        prediction = self.model.predict(x)
        return prediction[0]

    def predict_outcome(self, log):
        x = self.log_encoder.transform(log, for_training=False)
        prediction = self.model.predict(x)
        return prediction[1]

    def predict_next_time(self, log):
        x = self.log_encoder.transform(log, for_training=False)
        prediction = self.model.predict(x)
        return np.maximum(prediction[2].flatten() * self.log_encoder.time_scaling_divisor[0], 0)

    def predict_cycle_time(self, log):
        x = self.log_encoder.transform(log, for_training=False)
        prediction = self.model.predict(x)
        passed_time = np.array(
                [case[-1]["time:timestamp"].timestamp() - case[0]["time:timestamp"].timestamp() for case in log])
        return np.add(passed_time, np.maximum(prediction[3].flatten() * self.log_encoder.time_scaling_divisor[0], 0))

    def predict(self, log):
        x = self.log_encoder.transform(log, for_training=False)
        prediction = self.model.predict(x)
        prediction[1] = np.maximum(prediction[1].flatten() * self.log_encoder.time_scaling_divisor[0], 0)
        return prediction

    def _evaluate_raw(self, log):
        # Make predictions
        prefix_log = [case[0:prefix_length] for case in log for prefix_length in range(1, len(case) + 1)]
        predictions = self.predict(prefix_log)
        predicted_next_activities = np.argmax(predictions[0], axis=1)
        predicted_next_times = predictions[1] / 86400
        caseIDs = []
        prefix_lengths = []
        true_next_activities = []
        true_next_times = []
        for case in log:
            caseID = case.attributes["concept:name"]
            for prefix_length in range(1, len(case) + 1):
                caseIDs.append(caseID)
                prefix_lengths.append(prefix_length)
                true_next_activities.append(len(self.activities) if prefix_length == len(case) else self.activities.index(
                        case[prefix_length]["concept:name"]) if case[prefix_length]["concept:name"] in self.activities else -1)
                true_next_times.append(0 if prefix_length == len(case) else (case[prefix_length]["time:timestamp"].timestamp() -
                                                                             case[prefix_length - 1]["time:timestamp"].timestamp()) / 86400)

        # Generate DataFrame
        column_data = {"caseID": caseIDs, "prefix-length": prefix_lengths, "true-next-activity": true_next_activities,
                       "pred-next-activity": predicted_next_activities, "true-next-time": true_next_times,
                       "pred-next-time": predicted_next_times}
        columns = ["caseID", "prefix-length", "true-next-activity", "pred-next-activity", "true-next-time", "pred-next-time"]
        return pd.DataFrame(column_data, columns=columns)

    def evaluate(self, log, path, num_prefixes=8):
        # Generate raw predictions
        raw = self._evaluate_raw(log)
        raw.to_csv("./results/tapp_" + (self.log_encoder.text_encoder.name if self.log_encoder.text_encoder is not None else "-") +
                   "_" + (str(self.log_encoder.text_encoder.encoding_length) if self.log_encoder.text_encoder is not None else "0") +
                   path.replace("/", ""), encoding="utf-8", sep=",", index=False)
        # Compute metrics
        next_activity_acc_pre = [len(raw[(raw["pred-next-activity"] == raw["true-next-activity"]) & (
                    raw["prefix-length"] == prefix_length)]) / np.max([len(raw[raw["prefix-length"] == prefix_length]), 1])
                                 for prefix_length in range(1, num_prefixes + 1)]
        next_time_mae_pre = [
            mean_absolute_error(raw[raw["prefix-length"] == prefix_length]["true-next-time"].astype(float).to_numpy(),
                                raw[raw["prefix-length"] == prefix_length]["pred-next-time"].astype(
                                    float).to_numpy()).numpy() for prefix_length in range(1, num_prefixes + 1)]

        prefix_predictions = next_activity_acc_pre + next_time_mae_pre

        next_activity_y_true = raw[raw["prefix-length"] >= 1]["true-next-activity"].astype(float).to_numpy()
        next_activity_y_pred = raw[raw["prefix-length"] >= 1]["pred-next-activity"].astype(float).to_numpy()
        next_activity_acc = len(
                raw[(raw["pred-next-activity"] == raw["true-next-activity"]) & (raw["prefix-length"] >= 1)]) / np.max(
                [len(raw[raw["prefix-length"] >= 1]), 1])

        next_activity_f1_weighted = f1_score(next_activity_y_true, next_activity_y_pred, average='weighted')
        next_activity_f1_micro = f1_score(next_activity_y_true, next_activity_y_pred, average='micro')
        next_activity_f1_macro = f1_score(next_activity_y_true, next_activity_y_pred, average='macro')

        next_activity_precision_weighted = precision_score(next_activity_y_true, next_activity_y_pred, average='weighted')
        next_activity_precision_micro = precision_score(next_activity_y_true, next_activity_y_pred, average='micro')
        next_activity_precision_macro = precision_score(next_activity_y_true, next_activity_y_pred, average='macro')

        next_activity_recall_weighted = recall_score(next_activity_y_true, next_activity_y_pred, average='weighted')
        next_activity_recall_micro = recall_score(next_activity_y_true, next_activity_y_pred, average='micro')
        next_activity_recall_macro = recall_score(next_activity_y_true, next_activity_y_pred, average='macro')

        next_time_mae = mean_absolute_error(raw[raw["prefix-length"] >= 1]["true-next-time"].astype(float).to_numpy(),
                                            raw[raw["prefix-length"] >= 1]["pred-next-time"].astype(float).to_numpy()).numpy()
        next_time_mse = mean_squared_error(raw[raw["prefix-length"] >= 1]["true-next-time"].astype(float).to_numpy(),
                                           raw[raw["prefix-length"] >= 1]["pred-next-time"].astype(float).to_numpy()).numpy()
        next_time_rmse = math.sqrt(next_time_mse)

        if not os.path.exists("./results/" + path):
            prefix_columns = []
            for metric in ["naa_{}", "ntm_{}"]:
                for prefix in range(1, num_prefixes + 1):
                    prefix_columns.append(metric.format(prefix))
            columns = ["model", "timestamp", "num_layer", "num_shared_layer", "hidden_neurons", "advanced_time_attributes",
                       "data_attributes", "event_dim", "text_encoding", "text_dim",
                       "next_activity_acc",
                       "next_activity_f1_weighted", "next_activity_f1_micro", "next_activity_f1_macro",
                       "next_activity_precision_weighted", "next_activity_precision_micro", "next_activity_precision_macro",
                       "next_activity_recall_weighted", "next_activity_recall_micro", "next_activity_recall_macro",
                       "next_time_mae", "next_time_mse", "next_time_rmse"] + prefix_columns
            df = pd.DataFrame(columns=columns)
            df.to_csv("./results/" + path, encoding="utf-8", sep=",", index=False)
        df = pd.read_csv("./results/" + path, sep=",")

        df.loc[len(df)] = ["tapp", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                           self.num_shared_layer + self.num_specialized_layer, self.num_shared_layer, self.neurons_per_layer,
                           self.log_encoder.advanced_time_attributes,
                           self.log_encoder.data_attributes, self.log_encoder.feature_dim,
                           self.log_encoder.text_encoder.name if self.log_encoder.text_encoder is not None else "-",
                           self.log_encoder.text_encoder.encoding_length if self.log_encoder.text_encoder is not None else 0,
                           next_activity_acc,
                           next_activity_f1_weighted, next_activity_f1_micro, next_activity_f1_macro,
                           next_activity_precision_weighted, next_activity_precision_micro, next_activity_precision_macro,
                           next_activity_recall_weighted, next_activity_recall_micro, next_activity_recall_macro,
                           next_time_mae, next_time_mse, next_time_rmse
                           ] + prefix_predictions
        df.to_csv("./results/" + path, encoding="utf-8", sep=",", index=False)
        return df

    def _get_activity_label(self, index):
        return self.activities[index]


def _get_event_labels(log, attribute_name):
    return list(dict.fromkeys([event[attribute_name] for case in log for event in case])) if log else []
