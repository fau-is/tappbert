from datetime import datetime
import os
from .prediction_model import PredictionModel
from tensorflow.keras.losses import mean_absolute_error
import numpy as np
import pandas as pd


def _abstract_seq(seq, abstraction="sequence"):
    # Compute abstraction for a sequence
    if abstraction == "sequence":
        # Use tuples
        abstraction = tuple(seq)
    elif abstraction == "bag":
        # Use sorted tuples
        abstraction = tuple(sorted(seq))
    elif abstraction == "set":
        # Use frozen sets
        abstraction = frozenset(seq)
    else:
        # Default
        abstraction = tuple(seq)
    return abstraction


class TSModel(PredictionModel):

    def __init__(self):
        self.abstraction = None
        self.states = None
        self.horizon = 100
        self.transitions = None
        self.activities = []
        self.attribute = None
        self.remaining_time_anno = {}
        self.next_time_anno = {}
        self.next_activity_anno = {}
        self.outcome_anno = {}
        self.remaining_time_global = 0
        self.next_time_global = 0
        self.next_activity_global = None
        self.outcome_global = None
        self.test = None
        super().__init__()

    def fit(self, log, attribute="concept:name", horizon=3, abstraction="sequence"):
        self.attribute = attribute
        self.horizon = horizon
        self.abstraction = abstraction

        # Compute list of activities
        self.activities = list(dict.fromkeys([event[attribute] for case in log for event in case])) if log else []

        # Generate transition system by computing states and transitions
        self.states = [_abstract_seq([], abstraction=abstraction)]
        self.transitions = {}
        for case in log:
            trace = [self.activities.index(event[attribute]) for event in case]
            for i, event in enumerate(case):
                prev_state = _abstract_seq(trace[max(0, i - horizon):i], abstraction=abstraction)
                next_state = _abstract_seq(trace[max(0, i - horizon + 1): i + 1], abstraction=abstraction)
                transition = (prev_state, self.activities.index(event[attribute]), next_state)

                # Add state
                if next_state not in self.states:
                    self.states.append(next_state)
                    self.remaining_time_anno[next_state] = []
                    self.next_time_anno[next_state] = []
                    self.next_activity_anno[next_state] = []
                    self.outcome_anno[next_state] = []

                # Add transition
                if transition not in self.transitions:
                    self.transitions[transition] = 1
                else:
                    self.transitions[transition] += 1

                # Add measurements
                self.remaining_time_anno[next_state].append(case[-1]["time:timestamp"].timestamp() - event["time:timestamp"].timestamp())
                self.next_time_anno[next_state].append((case[i+1]["time:timestamp"].timestamp() - event["time:timestamp"].timestamp()) if i < (len(case) - 1) else 0)
                self.next_activity_anno[next_state].append(self.activities.index(case[i+1][self.attribute]) if i < len(case) - 1 else len(self.activities))
                self.outcome_anno[next_state].append(self.activities.index(case[-1][self.attribute]))

        self.test = self.next_time_anno

        # Compute global distributions
        self.remaining_time_global = np.median([time for state in self.states if state in self.remaining_time_anno for time in self.remaining_time_anno[state]])
        self.next_time_global = np.median([time for state in self.states if state in self.next_time_anno for time in self.next_time_anno[state]])
        next_activity_list = [self.activities.index(event[attribute]) for case in log for event in case if event is not case[0]] + [len(self.activities) for _ in log]
        self.next_activity_global = np.array([next_activity_list.count(activity) for activity in range(len(self.activities) + 1)])
        outcome_list = [self.activities.index(case[-1][attribute]) for case in log]
        self.outcome_global = np.array([outcome_list.count(activity) for activity in range(len(self.activities))])

        # Compute annotations
        for state in self.states:
            if state in self.remaining_time_anno:
                self.remaining_time_anno[state] = np.median(self.remaining_time_anno[state])
                self.next_time_anno[state] = np.median(self.next_time_anno[state])
                self.next_activity_anno[state] = np.array([(self.next_activity_anno[state].count(activity) / len(self.next_activity_anno[state])) for activity in range(len(self.activities) + 1)])
                self.outcome_anno[state] = np.array([(self.outcome_anno[state].count(activity) / len(self.outcome_anno[state])) for activity in range(len(self.activities))])

    def predict_next_activity(self, log):
        predictions = []
        for case in log:
            trace = [self.activities.index(event[self.attribute]) for event in case if event[self.attribute] in self.activities]
            state = _abstract_seq(trace[max(0, len(trace) - self.horizon):], abstraction=self.abstraction)
            if state in self.next_activity_anno:
                predictions.append(self.next_activity_anno[state])
            else:
                predictions.append(self.next_activity_global)
        return np.array(predictions)

    def predict_outcome(self, log):
        predictions = []
        for case in log:
            trace = [self.activities.index(event[self.attribute]) for event in case if event[self.attribute] in self.activities]
            state = _abstract_seq(trace[max(0, len(trace) - self.horizon):], abstraction=self.abstraction)
            if state in self.outcome_anno:
                predictions.append(self.outcome_anno[state])
            else:
                predictions.append(self.outcome_global)
        return np.array(predictions)

    def predict_next_time(self, log):
        predictions = []
        for case in log:
            trace = [self.activities.index(event[self.attribute]) for event in case if event[self.attribute] in self.activities]
            state = _abstract_seq(trace[max(0, len(trace) - self.horizon):], abstraction=self.abstraction)
            if state in self.next_time_anno:
                predictions.append(self.next_time_anno[state])
            else:
                predictions.append(self.next_time_global)
        return np.array(predictions)

    def predict_cycle_time(self, log):
        predictions = []
        for case in log:
            trace = [self.activities.index(event[self.attribute]) for event in case if event[self.attribute] in self.activities]
            state = _abstract_seq(trace[max(0, len(trace) - self.horizon):], abstraction=self.abstraction)
            if state in self.remaining_time_anno:
                predictions.append(case[-1]["time:timestamp"].timestamp() - case[0]["time:timestamp"].timestamp() + self.remaining_time_anno[state])
            else:
                predictions.append(case[-1]["time:timestamp"].timestamp() - case[0]["time:timestamp"].timestamp() + self.remaining_time_global)
        return np.array(predictions)

    def predict(self, log):
        return [self.predict_next_activity(log), self.predict_outcome(log), self.predict_next_time(log), self.predict_cycle_time(log)]

    def _evaluate_raw(self, log):
        # Make predictions
        prefix_log = [case[0:prefix_length] for case in log for prefix_length in range(1, len(case) + 1)]
        predictions = self.predict(prefix_log)
        predicted_next_activities = np.argmax(predictions[0], axis=1)
        predicted_case_outcomes = np.argmax(predictions[1], axis=1)
        predicted_next_times = predictions[2] / 86400
        predicted_cycle_times = predictions[3] / 86400
        caseIDs = []
        prefix_lengths = []
        true_next_activities = []
        true_case_outcomes = []
        true_next_times = []
        true_cycle_times = []
        for case in log:
            caseID = case.attributes["concept:name"]
            for prefix_length in range(1, len(case) + 1):
                caseIDs.append(caseID)
                prefix_lengths.append(prefix_length)

                true_next_activities.append(len(self.activities) if prefix_length == len(case) else self.activities.index(case[prefix_length]["concept:name"]) if case[prefix_length]["concept:name"] in self.activities else -1)
                true_case_outcomes.append(self.activities.index(case[-1]["concept:name"]) if case[-1]["concept:name"] in self.activities else -1)
                true_next_times.append(0 if prefix_length == len(case) else (case[prefix_length]["time:timestamp"].timestamp() - case[prefix_length - 1]["time:timestamp"].timestamp()) / 86400)
                true_cycle_times.append((case[-1]["time:timestamp"].timestamp() - case[0]["time:timestamp"].timestamp()) / 86400)

        # Generate DataFrame
        column_data = {"caseID": caseIDs, "prefix-length": prefix_lengths, "true-next-activity": true_next_activities, "pred-next-activity": predicted_next_activities, "true-outcome": true_case_outcomes, "pred-outcome": predicted_case_outcomes, "true-next-time": true_next_times,
            "pred-next-time": predicted_next_times, "true-cycle-time": true_cycle_times, "pred-cylce-time": predicted_cycle_times}
        columns = ["caseID", "prefix-length", "true-next-activity", "pred-next-activity", "true-next-time", "pred-next-time", "true-outcome", "pred-outcome", "true-cycle-time", "pred-cylce-time"]
        return pd.DataFrame(column_data, columns=columns)

    def evaluate(self, log, path, num_prefixes=8):
        # Generate raw predictions
        raw = self._evaluate_raw(log)
        raw.to_csv("ts_" + self.abstraction + "_" + str(self.horizon) + path.replace("/",""), encoding="utf-8", sep=",", index=False)
        # Compute metrics
        next_activity_acc = len(raw[(raw["pred-next-activity"] == raw["true-next-activity"]) & (raw["prefix-length"] >= 1)]) / np.max([len(raw[raw["prefix-length"] >= 1]), 1])
        next_time_mae = mean_absolute_error(raw[raw["prefix-length"] >= 1]["true-next-time"].astype(float).to_numpy(), raw[raw["prefix-length"] >= 1]["pred-next-time"].astype(float).to_numpy()).numpy()
        outcome_acc = len(raw[(raw["pred-outcome"] == raw["true-outcome"]) & (raw["prefix-length"] >= 1)]) / np.max([len(raw[raw["prefix-length"] >= 1]), 1])
        cycle_time_mae = mean_absolute_error(raw[raw["prefix-length"] >= 1]["true-cycle-time"].astype(float).to_numpy(), raw[raw["prefix-length"] >= 1]["pred-cylce-time"].astype(float).to_numpy()).numpy()

        next_activity_acc_pre = [len(raw[(raw["pred-next-activity"] == raw["true-next-activity"]) & (raw["prefix-length"] == prefix_length)]) / np.max([len(raw[raw["prefix-length"] == prefix_length]), 1]) for prefix_length in range(1, num_prefixes + 1)]
        next_time_mae_pre = [mean_absolute_error(raw[raw["prefix-length"] == prefix_length]["true-next-time"].astype(float).to_numpy(), raw[raw["prefix-length"] == prefix_length]["pred-next-time"].astype(float).to_numpy()).numpy() for prefix_length in range(1, num_prefixes + 1)]
        outcome_acc_pre = [len(raw[(raw["pred-outcome"] == raw["true-outcome"]) & (raw["prefix-length"] == prefix_length)]) / np.max([len(raw[raw["prefix-length"] == prefix_length]), 1]) for prefix_length in range(1, num_prefixes + 1)]
        cycle_time_mae_pre = [mean_absolute_error(raw[raw["prefix-length"] == prefix_length]["true-cycle-time"].astype(float).to_numpy(), raw[raw["prefix-length"] == prefix_length]["pred-cylce-time"].astype(float).to_numpy()).numpy() for prefix_length in range(1, num_prefixes + 1)]

        prefix_predictions = next_activity_acc_pre + next_time_mae_pre + outcome_acc_pre + cycle_time_mae_pre

        if not os.path.exists(path):
            prefix_columns = []
            for metric in ["naa_{}", "ntm_{}", "oa_{}", "ctm_{}"]:
                for prefix in range(1, num_prefixes + 1):
                    prefix_columns.append(metric.format(prefix))
            columns = ["model", "timestamp", "num_layer", "num_shared_layer", "hidden_neurons", "advanced_time_attributes", "data_attributes", "event_dim", "text_encoding", "text_dim", "next_activity_acc", "next_time_mae", "outcome_acc", "cycle_time_mae"] + prefix_columns
            df = pd.DataFrame(columns=columns)
            df.to_csv(path, encoding="utf-8", sep=",", index=False)
        df = pd.read_csv(path, sep=",")

        df.loc[len(df)] = ["ts", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), "-", "-", "-", "-", "-", "-", self.abstraction, self.horizon, next_activity_acc, next_time_mae, outcome_acc, cycle_time_mae] + prefix_predictions
        df.to_csv(path, encoding="utf-8", sep=",", index=False)
        return df
