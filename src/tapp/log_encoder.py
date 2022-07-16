import numpy as np
from abc import ABC, abstractmethod
from tapp.text_encoder import BERTbaseFineTunedNextActivityTextEncoder
from tapp.text_encoder import BERTbaseFineTunedNextTimeTextEncoder
from tapp.text_encoder import BERTbaseFineTunedNextActivityAndTimeTextEncoder


class Encoder(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, docs):
        pass

    @abstractmethod
    def transform(self, docs):
        pass


class LogEncoder(Encoder):

    def __init__(self, text_encoder=None, advanced_time_attributes=True, text_base_for_training='event'):
        self.text_encoder = text_encoder
        self.activities = []
        self.data_attributes = []
        self.text_attribute = None
        self.categorical_attributes = []
        self.categorical_attributes_values = []
        self.numerical_attributes = []
        self.numerical_divisor = []
        self.event_dim = 0
        self.feature_dim = 0
        self.advanced_time_attributes = advanced_time_attributes
        self.time_scaling_divisor = [1, 1, 1]
        self.process_start_time = 0
        self.text_base_for_training = text_base_for_training
        super().__init__()

    def fit(self, log, activities=None, data_attributes=None, text_attribute=None):
        # Fit encoder to log
        self.activities = activities
        self.data_attributes = data_attributes
        self.text_attribute = text_attribute
        self.categorical_attributes = list(filter(lambda attribute: not _is_numerical_attribute(log, attribute), self.data_attributes))
        self.categorical_attributes_values = [_get_event_labels(log, attribute) for attribute in self.categorical_attributes]
        self.numerical_attributes = list(filter(lambda attribute: _is_numerical_attribute(log, attribute), self.data_attributes))
        self.numerical_divisor = [np.max([event[attribute].timestamp() for case in log for event in case]) for attribute in self.numerical_attributes]
        self.process_start_time = np.min([event["time:timestamp"].timestamp() for case in log for event in case])

        # Scaling divisors for time related features to achieve values between 0 and 1
        time_between_events_max = np.max(
            [event["time:timestamp"].timestamp() - case[event_index - 1]["time:timestamp"].timestamp() for case in
             log for event_index, event in enumerate(case) if event_index > 0])
        self.time_scaling_divisor = [time_between_events_max, 86400, 604800]  # 86400 = 24 * 60 * 60, 604800 = 7 * 24 * 60 * 60

        # Event dimension: Maximum number of events in a case
        self.event_dim = _get_max_case_length(log)

        # Feature dimension: Encoding size of an event
        activity_encoding_length = len(self.activities)
        time_encoding_length = 6 if self.advanced_time_attributes else 2
        categorical_attributes_encoding_length = sum([len(values) for values in self.categorical_attributes_values])
        numerical_attributes_encoding_length = len(self.numerical_attributes)
        text_encoding_length = self.text_encoder.encoding_length if self.text_encoder is not None and self.text_attribute is not None else 0
        if isinstance(self.text_encoder, BERTbaseFineTunedNextActivityAndTimeTextEncoder):
            self.feature_dim = self.feature_dim = activity_encoding_length + time_encoding_length + categorical_attributes_encoding_length + numerical_attributes_encoding_length + 2 * text_encoding_length
        else:
            self.feature_dim = self.feature_dim = activity_encoding_length + time_encoding_length + categorical_attributes_encoding_length + numerical_attributes_encoding_length + text_encoding_length

        # Train text encoder
        if self.text_encoder is not None and self.text_attribute is not None:
            # collect texts / documents and labels for BERT fine-tuning (labels are indices, API: "should be in [0, ..., config.num_labels - 1]")
            event_docs = [event[self.text_attribute] for case in log for event in case if self.text_attribute in event]
            event_next_activities = []
            event_next_times = []

            prefix_docs = []
            prefix_next_activities = []
            prefix_next_times = []

            for case in log:
                for event_i, event in enumerate(case):
                    if self.text_attribute in event:
                        # next activity
                        next_event_i = event_i + 1
                        if next_event_i == len(case):
                            # case terminated
                            event_next_activities.append(len(self.activities))
                            event_next_times.append(0)
                        else:
                            # case ongoing
                            event_next_activities.append(self.activities.index(case[next_event_i]["concept:name"]))
                            event_next_times.append((case[next_event_i]["time:timestamp"].timestamp() -
                                                     case[event_i]["time:timestamp"].timestamp()) /
                                                    self.time_scaling_divisor[0])

                case_event_idx_docs = [(event_i, event[self.text_attribute]) for event_i, event in enumerate(case) if self.text_attribute in event]

                # skip case with only one document (already in event_docs)
                if len(case_event_idx_docs) <= 1:
                    continue
                else:
                    case_docs = ''
                    case_next_activities = []
                    case_next_times = []

                    # concat events' texts
                    for _, doc in case_event_idx_docs:
                        case_docs += ' ' + doc

                    # prefix next activity
                    next_event_i = case_event_idx_docs[-1][0]
                    if next_event_i == len(case):
                        # case terminated
                        case_next_activities.append(len(self.activities))
                        case_next_times.append(0)
                    else:
                        # case ongoing
                        case_next_activities.append(self.activities.index(case[next_event_i]["concept:name"]))
                        case_next_times.append((case[next_event_i]["time:timestamp"].timestamp() - case[event_i][
                            "time:timestamp"].timestamp()) / self.time_scaling_divisor[0])

                    prefix_docs.append(case_docs)
                    prefix_next_activities.extend(case_next_activities)
                    prefix_next_times.extend(case_next_times)

            if self.text_base_for_training == 'event':
                docs = event_docs
                next_activities = event_next_activities
                next_times = event_next_times
            elif self.text_base_for_training == 'prefix':
                docs = event_docs + prefix_docs
                next_activities = event_next_activities + prefix_next_activities
                next_times = event_next_times + prefix_next_times

            # fine-tune BERT on next activity prediction (Sequence Classification)
            if isinstance(self.text_encoder, BERTbaseFineTunedNextActivityTextEncoder):
                self.text_encoder.fit(docs, np.array(next_activities))

            # fine-tune BERT on next event time prediction (Sequence Regression)
            elif isinstance(self.text_encoder, BERTbaseFineTunedNextTimeTextEncoder):
                self.text_encoder.fit(docs, np.array(next_times))

            elif isinstance(self.text_encoder, BERTbaseFineTunedNextActivityAndTimeTextEncoder):
                self.text_encoder.fit(docs, np.array(next_activities))

            # other text model
            else:
                self.text_encoder.fit(docs)

    def transform(self, log, for_training=True):
        case_dim = np.sum([len(case) for case in log]) if for_training else len(log)

        # Prepare input and output vectors/matrices
        x = np.zeros((case_dim, self.event_dim, self.feature_dim))
        if for_training:
            y_next_act = np.zeros((case_dim, len(self.activities) + 1))
            y_next_time = np.zeros(case_dim)

        # Encode traces and prefix traces
        trace_dim_index = 0
        for case in log:
            case_start_time = case[0]["time:timestamp"].timestamp()
            # For training: Encode all prefixes. For predicting: Encode given prefix only
            prefix_lengths = range(1, len(case) + 1) if for_training else range(len(case), len(case) + 1)
            for prefix_length in prefix_lengths:
                # Encode the (prefix-)trace
                previous_event_time = case_start_time
                # Post padding of event sequences
                padding = self.event_dim - prefix_length
                for event_index, event in enumerate(case):

                    if event_index <= prefix_length - 1:
                        # Encode activity
                        if event["concept:name"] in self.activities:
                            x[trace_dim_index][padding+event_index][self.activities.index(event["concept:name"])] = 1
                        offset = len(self.activities)

                        # Encode time attributes
                        event_time = event["time:timestamp"]
                        # Seconds since previous event
                        x[trace_dim_index][padding+event_index][offset + 1] = (event_time.timestamp() - previous_event_time)/self.time_scaling_divisor[0]
                        # Encode additional time attributes if option is chosen
                        if self.advanced_time_attributes:
                            # Seconds since midnight
                            x[trace_dim_index][padding+event_index][offset + 2] = (event_time.hour * 3600 + event_time.second)/self.time_scaling_divisor[1]
                            # Seconds since last Monday
                            x[trace_dim_index][padding+event_index][offset + 3] = (event_time.weekday() * 86400 + event_time.hour * 3600 + event_time.second)/self.time_scaling_divisor[2]
                            offset += 3
                        else:
                            offset += 1

                        previous_event_time = event_time.timestamp()

                        # Encode categorical attributes
                        for attribute_index, attribute in enumerate(self.categorical_attributes):
                            if event[attribute] in self.categorical_attributes_values[attribute_index]:
                                x[trace_dim_index][padding+event_index][offset + self.categorical_attributes_values[attribute_index].index(event[attribute])] = 1
                            offset += len(self.categorical_attributes_values[attribute_index])

                        # Encode numerical attributes
                        for attribute_index, attribute in enumerate(self.numerical_attributes):
                            x[trace_dim_index][padding+event_index][offset] = float(event[attribute])/self.numerical_divisor[attribute_index]
                            offset += 1

                        # Encode textual attribute
                        if self.text_encoder is not None and self.text_attribute is not None and self.text_attribute in event:
                            if isinstance(self.text_encoder, BERTbaseFineTunedNextActivityAndTimeTextEncoder):
                                text_vectors_act, text_vectors_time = self.text_encoder.transform([event[self.text_attribute]])
                                x[trace_dim_index][padding + event_index][offset:offset + len(text_vectors_act[0])] = \
                                text_vectors_act[0]
                                x[trace_dim_index][padding + event_index][offset:offset + len(text_vectors_time[0])] = \
                                    text_vectors_time[0]
                                offset += 2* self.text_encoder.encoding_length
                            else:
                                text_vectors = self.text_encoder.transform([event[self.text_attribute]])
                                x[trace_dim_index][padding+event_index][offset:offset+len(text_vectors[0])] = text_vectors[0]
                                offset += self.text_encoder.encoding_length

                # Set activity and time (since case start) of next event as target
                if for_training:
                    if prefix_length == len(case):
                        # Case 1: Set <Process end> as next activity target
                        y_next_act[trace_dim_index][len(self.activities)] = 1
                        y_next_time[trace_dim_index] = 0
                    else:
                        # Case 2: Set next activity as target
                        y_next_act[trace_dim_index][self.activities.index(case[prefix_length]["concept:name"])] = 1
                        y_next_time[trace_dim_index] = (case[prefix_length]["time:timestamp"].timestamp() - case[prefix_length - 1]["time:timestamp"].timestamp()) / self.time_scaling_divisor[0]

                # Increase index for next (prefix-)trace
                trace_dim_index += 1

        if for_training:
            return x, y_next_act, y_next_time
        else:
            return x


def _get_event_labels(log, attribute_name):
    return list(dict.fromkeys([event[attribute_name] for case in log for event in case])) if log else []


def _is_numerical(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def _is_numerical_attribute(log, attribute):
    return _is_numerical(log[0][0][attribute])


def _get_max_case_length(log):
    return max([len(case) for case in log]) if log else 0
