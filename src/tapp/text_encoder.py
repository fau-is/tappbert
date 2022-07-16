import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import TaggedDocument
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from abc import ABC, abstractmethod
import datetime
import pathlib
import time
import pandas as pd
import torch
import transformers
from matplotlib import pyplot as plt
import seaborn as sns
from gensim.models import Doc2Vec
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, BertForPreTraining, BertConfig
from transformers import TextDatasetForNextSentencePrediction, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from tokenizers import BertWordPieceTokenizer
import os


class TextEncoder(ABC):

    def __init__(self, language="english", encoding_length=50):
        self.language = language

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet")
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

        self.encoding_length = encoding_length
        self.stop_words = set(stopwords.words(language))
        self.stop_words.remove("not")
        self.stop_words.remove("no")
        self.lemmatizer = WordNetLemmatizer() if language == "english" else None
        self.stemmer = SnowballStemmer(language)

        super().__init__()

    def preprocess_docs(self, docs, as_list=True):
        docs_preprocessed = []
        for doc in docs:
            doc = doc.lower()
            words = word_tokenize(doc, language=self.language)
            if self.lemmatizer is not None:
                words_processed = [self.lemmatizer.lemmatize(word) for word in words if
                                   word not in self.stop_words and word.isalpha()]
            else:
                words_processed = [self.stemmer.stem(word) for word in words if
                                   word not in self.stop_words and word.isalpha()]
            if as_list:
                docs_preprocessed.append(words_processed)
            else:
                docs_preprocessed.append(" ".join(words_processed))
        return docs_preprocessed

    @abstractmethod
    def fit(self, docs):
        pass

    @abstractmethod
    def transform(self, docs):
        pass


class BoWTextEncoder(TextEncoder):

    def __init__(self, language="english", encoding_length=100):
        self.name = "BoW"
        self.vectorizer = None
        super().__init__(language=language, encoding_length=encoding_length)

    def fit(self, docs):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=self.encoding_length, analyzer='word', norm="l2")
        self.vectorizer.fit(self.preprocess_docs(docs, as_list=False))
        return self

    def transform(self, docs):
        return self.vectorizer.transform(self.preprocess_docs(docs, as_list=False)).toarray()


class BoNGTextEncoder(TextEncoder):

    def __init__(self, language="english", encoding_length=100, n=2):
        self.name = "BoNG"
        self.n = n
        self.vectorizer = None
        super().__init__(language=language, encoding_length=encoding_length)

    def fit(self, docs):
        self.vectorizer = TfidfVectorizer(ngram_range=(self.n, self.n), max_features=self.encoding_length, analyzer='word',
                                          norm="l2")
        self.vectorizer.fit(self.preprocess_docs(docs, as_list=False))
        return self

    def transform(self, docs):
        return self.vectorizer.transform(self.preprocess_docs(docs, as_list=False)).toarray()


class PVTextEncoder(TextEncoder):

    def __init__(self, language="english", encoding_length=20, epochs=15, min_count=2):
        self.name = "PV"
        self.epochs = epochs
        self.min_count = min_count
        self.model = None
        super().__init__(language=language, encoding_length=encoding_length)

    def fit(self, docs):
        docs = self.preprocess_docs(docs)

        tagged_docs = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(docs)]

        self.model = Doc2Vec(dm=1, vector_size=self.encoding_length, min_count=self.min_count, window=8)
        self.model.build_vocab(tagged_docs)

        self.model.train(utils.shuffle(tagged_docs), total_examples=len(tagged_docs), epochs=self.epochs)
        self.model.save(f"pv_{self.encoding_length}_model")
        print("Saved Doc2Vec model!")
        return self

    def transform(self, docs):
        docs = self.preprocess_docs(docs)
        return np.array([self.model.infer_vector(doc) for doc in docs])


class LDATextEncoder(TextEncoder):

    def __init__(self, language="english", encoding_length=20):
        self.name = "LDA"
        self.model = None
        self.num_topics = encoding_length
        self.dictionary = None
        super().__init__(language=language, encoding_length=encoding_length)

    def fit(self, docs):
        docs = self.preprocess_docs(docs)
        self.dictionary = Dictionary(docs)
        corpus = [self.dictionary.doc2bow(doc) for doc in docs]
        self.model = LdaModel(corpus, id2word=self.dictionary, num_topics=self.num_topics, minimum_probability=0.0)
        return self

    def transform(self, docs):
        docs = self.preprocess_docs(docs)
        docs = [self.dictionary.doc2bow(doc) for doc in docs]
        return np.array([self.model[doc] for doc in docs])[:, :, 1]


class BERTbaseTextEncoder(TextEncoder):

    def __init__(self, language="english", encoding_length=20, max_sentence_length_buffer=12, embedding_level="cls"):
        self.name = "BERTbase"
        self.model = None
        self.tokenizer = None
        self.max_sentence_length = 0
        self.max_sentence_length_buffer = max_sentence_length_buffer
        self.embedding_level = embedding_level
        super().__init__(language=language, encoding_length=encoding_length)

    def fit(self, docs, **kwargs):
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return self

    def transform(self, docs):
        # docs = self.preprocess_docs(docs, as_list=False)
        list_embeddings = []
        for doc in docs:
            tokenized_text, tokens_tensor, segments_tensor = self.bert_text_preparation(doc)
            list_embeddings.append(self.get_bert_embeddings(tokens_tensor, segments_tensor,
                                                            embedding_level=self.embedding_level))
        return np.squeeze(np.array(list_embeddings), axis=0)

    def bert_text_preparation(self, text):
        """Preparing the input for BERT

        Takes a string argument and performs pre-processing like adding special tokens, tokenization, tokens to ids,
        and tokens to segment ids. All tokens are mapped to segment id = 1.

        Args:
            text (str): Text to be converted
            tokenizer (obj): Tokenizer object to convert text into BERT-readable tokens and ids

        Returns:
            list: List of BERT-readable tokens
            obj: Torch tensor with token ids
            obj: Torch tensor segment ids

        """
        # 1. Add special tokens to start and end of sentence
        # [SEP] token at the end of each sentence
        # [CLS] token at beginning of each sentence; the final hidden state corresponding to this token is used as the
        # aggregate sequence representation for classification tasks
        marked_text = "[CLS] " + text + " [SEP]"  # here, [CLS] is ignored
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokenized_text, tokens_tensor, segments_tensors

    def get_bert_embeddings(self, tokens_tensor, segments_tensors, embedding_level="word"):
        """Get embeddings from an embedding model

        Args:
            tokens_tensor (obj): Torch tensor size [n_tokens] with token ids for each token in text
            segments_tensors (obj): Torch tensor size [n_tokens] with segment ids for each token in text
            model (obj): Embedding model to generate embeddings from token and segment ids

        Returns:
            obj: Numpy array of floats of size [n_tokens, n_embedding_dimensions] containing embeddings for each token

        """

        # Gradient calculation id disabled, model is in inference mode
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            # 'hidden states' is a tensor with shape [layers, batch, token, feature/hidden unit]
            # layers is 13 (= 12 BERT layers + 1 input embeddings)
            hidden_states = outputs[2]
            # Removing the first hidden state (the first state is the input state)
            hidden_states = hidden_states[1:]

        # Concatenate the tensors for all layers
        token_embeddings = torch.stack(hidden_states, dim=0)

        if embedding_level == "word":
            # Remove batches dimension
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            # Swap dimensions 0 and 1.
            token_embeddings = token_embeddings.permute(1, 0, 2)

            token_embeddings_cat = []
            token_embeddings_sum = []

            # Skip embedding of [CLS] and [SEP] token
            for token in token_embeddings[1:-1]:
                token_embedding_cat = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
                token_embeddings_cat.append(token_embedding_cat)

                token_embedding_sum = torch.sum(token[-4:], dim=0)
                token_embeddings_sum.append(token_embedding_sum)

            # Sum over all token embeddings
            final_embeddings = torch.stack(token_embeddings_sum, dim=0).sum(dim=0)
            return torch.unsqueeze(final_embeddings, dim=0).numpy()

        if embedding_level == "sentence":
            # Take second to last hidden layer of each token
            token_embeddings = hidden_states[-2][0]
            # Calculate the average of all token vectors
            sentence_embedding = torch.mean(token_embeddings[1:-1], dim=0)
            return torch.unsqueeze(sentence_embedding, dim=0).numpy()

        if embedding_level == "cls":
            # [CLS] embeddings are the aggregate sequence representation for classification tasks
            cls_embeddings = []
            for i_layer in range(len(hidden_states)):
                cls_embeddings.append(hidden_states[i_layer][:, 0, :])

            # Sum over all cls embeddings
            final_embeddings = torch.stack(cls_embeddings, dim=0).sum(dim=0)

        if embedding_level == "cls_last":
            # [CLS] embeddings are the aggregate sequence representation for classification tasks
            cls_embeddings = []
            for i_layer in range(len(hidden_states)):
                cls_embeddings.append(hidden_states[i_layer][:, 0, :])
            # Sum over all cls embeddings
            final_embeddings = torch.stack(cls_embeddings, dim=0)[-1]

        return final_embeddings.numpy()


class BERTbaseFineTunedNextActivityTextEncoder(TextEncoder):

    def __init__(self, language="english", encoding_length=20, max_sentence_length_buffer=12, batch_size=32,
                 lr=2e-5, eps=1e-8, epochs=2, hidden_states_representations='cls', text_base_for_training='event',
                 model_dir="./models"):
        self.name = "BERTbaseFineTunedNextActivity"
        self.model = None
        self.tokenizer = None
        self.max_sentence_length = 0
        self.max_sentence_length_buffer = max_sentence_length_buffer
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.epochs = epochs
        self.hidden_states_representations = hidden_states_representations
        self.text_base_for_training = text_base_for_training
        self.name = f"{self.name}_{text_base_for_training}_{hidden_states_representations}_e{epochs}_lr{lr}"
        self.model_dir = model_dir
        self.device = None
        super().__init__(language=language, encoding_length=encoding_length)

    def fit(self, docs, labels):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

        self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                # num_labels=len(labels),
                num_labels=len(set(labels)),
                output_attentions=False,  # Return attention weights
                output_hidden_states=True  # Return all hidden states
        )
        self.print_model_specifications()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Determine max. sentence length (for padding / truncating)
        for doc in docs:
            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = self.tokenizer.encode(doc, add_special_tokens=True)
            # Update the maximum sentence length.
            self.max_sentence_length = max(self.max_sentence_length, len(input_ids))

        if self.max_sentence_length > 512:
            self.max_sentence_length = 512
        # Add buffer in case there are some longer test sentences
        self.max_sentence_length += self.max_sentence_length_buffer

        docs = self.preprocess_docs(docs, as_list=False)
        input_ids, attention_masks, labels = self.bert_input_preparation(docs, labels, for_training=True)
        train_dataloader, validation_dataloader = self.get_dataloader(input_ids, attention_masks, labels)
        self.bert_fine_tuning(train_dataloader, validation_dataloader)

        return self

    def transform(self, docs):
        docs = self.preprocess_docs(docs, as_list=False)
        input_ids, attention_masks, _ = self.bert_input_preparation(docs, labels=None, for_training=False)
        embeddings = self.get_bert_embeddings(input_ids, attention_masks,
                                              hidden_states_representations=self.hidden_states_representations)
        return embeddings

    def bert_input_preparation(self, docs, labels=None, for_training=True):
        """Preparing the input for BERT"""

        input_ids = []
        attention_masks = []

        for doc in docs:
            encoded_dict = self.tokenizer.encode_plus(
                    doc,
                    add_special_tokens=True,  # Add [CLS] and [SEP]
                    max_length=self.max_sentence_length,  # Pad or truncate sentence
                    # pad_to_max_length=True,
                    # truncation=True,
                    padding='max_length',
                    return_attention_mask=True,  # Construct attention masks
                    return_tensors='pt'  # Return pytorch tensors
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        if for_training:
            labels = torch.tensor(labels, dtype=torch.long)

        return input_ids, attention_masks, labels

    def get_dataloader(self, input_ids, attention_masks, labels=None, for_training=True):

        if for_training:
            # 90-10 Train-validation split
            dataset = TensorDataset(input_ids, attention_masks, labels)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size

            # Divide dataset by randomly selecting samples
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            print('{:>5,} training samples'.format(train_size))
            print('{:>5,} validation samples'.format(val_size))

            # Take training samples in random order
            train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=self.batch_size)
            # For validation the order doesn't matter, so just take them sequentially
            validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                               batch_size=self.batch_size)
            return train_dataloader, validation_dataloader

        else:
            prediction_data = TensorDataset(input_ids, attention_masks, labels)
            prediction_sampler = SequentialSampler(prediction_data)
            prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.batch_size)
            return prediction_dataloader

    def bert_fine_tuning(self, train_dataloader, validation_dataloader):

        print("")
        print("Fine-tuning BERT...")

        model_path = f"{self.model_dir}/{self.name}"
        if pathlib.Path(model_path).exists():
            # below for testing purposes: simply load previously fine-tuned model
            self.model = torch.load(model_path)
            print("Existing fine-tuned model loaded.")
            return

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # 'W' stands for 'Weight Decay fix" ??
        optimizer = transformers.AdamW(self.model.parameters(), lr=self.lr, eps=self.eps)
        # Total number of training steps is [number of batches] x [number of epochs]
        total_steps = len(train_dataloader) * self.epochs
        # Create learning rate scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                                 num_training_steps=total_steps)

        # # Set seed for reproducible results
        # seed_val = 0
        # random.seed(seed_val)
        # np.random.seed(seed_val)
        # torch.manual_seed(seed_val)
        # torch.cuda.manual_seed_all(seed_val)

        training_stats = []
        total_t0 = time.time()

        for epoch_i in range(0, self.epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 20 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # are given and what flags are set. For our usage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                print(b_input_ids.size(), b_input_mask.size(), b_labels.size())
                loss, logits, _ = self.model(b_input_ids,
                                             token_type_ids=None,
                                             attention_mask=b_input_mask,
                                             labels=b_labels,
                                             return_dict=False)

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
            self.model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    (loss, logits, _) = self.model(b_input_ids,
                                                   token_type_ids=None,
                                                   attention_mask=b_input_mask,
                                                   labels=b_labels,
                                                   return_dict=False)

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and accumulate it over all batches.
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                    {
                        'epoch': epoch_i + 1,
                        'Training Loss': avg_train_loss,
                        'Valid. Loss': avg_val_loss,
                        'Valid. Accur.': avg_val_accuracy,
                        'Training Time': training_time,
                        'Validation Time': validation_time
                    }
            )

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # save fine-tuned model
        torch.save(self.model, model_path)

        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time() - total_t0)))
        self.print_training_summary(training_stats)

    def get_bert_embeddings(self, input_ids, attention_masks, hidden_states_representations='cls'):

        # Gradient calculation id disabled, model is in inference mode
        with torch.no_grad():
            # BertForSequenceClassification is a wrapper that consists of two parts:
            # (1) BERT model (attribute 'bert') and (2) a classifier (attribute 'classifier').
            # here, call the BERT model directly to get the hidden states of the model
            outputs = self.model.bert(input_ids, attention_masks)
            last_hidden_states = outputs[0]
            cls_hidden_state = last_hidden_states[:, 0, :]
            hidden_states = outputs[2]

        if hidden_states_representations == "cls":
            embeddings = cls_hidden_state.numpy()
        if hidden_states_representations == "last_4_mean":
            embeddings = torch.mean(torch.stack(hidden_states[-4:], dim=0)[:, :, 0, :], dim=0).numpy()
        elif hidden_states_representations == "last_4_sum":
            sum_last_4_layers = torch.stack(hidden_states[-4:]).sum(0)
            embeddings = sum_last_4_layers.numpy()
        elif hidden_states_representations == "last_4_cat":
            concat_last_4_layers = torch.stack(hidden_states[-4:], dim=0)
            embeddings = concat_last_4_layers.numpy()

        return embeddings

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(self, elapsed):
        # Returns elapsed time, takes a time in seconds and returns a string hh:mm:ss
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def print_model_specifications(self):
        # Get all of the model's parameters as a list of tuples.
        params = list(self.model.named_parameters())

        print('The BERT model has {:} different named parameters.\n'.format(len(params)))

        print('==== Embedding Layer ====\n')

        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== First Transformer ====\n')

        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')

        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    def print_training_summary(self, training_stats):

        # Display floats with two decimal places.
        pd.set_option('precision', 2)

        # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=training_stats)

        # Use the 'epoch' as the row index.
        df_stats = df_stats.set_index('epoch')

        # A hack to force the column headers to wrap.
        # df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

        # Display the table
        df_stats

        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        # Plot the learning curve.
        plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
        plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

        # Label the plot.
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks([1, 2, 3, 4])

        plt.show()

    def evaluate_finetuned_model(self, prediction_dataloader):

        # Prediction on test set

        print('Predicting labels for {:,} test sentences...'.format(len(prediction_dataloader)))

        # Put model in evaluation mode
        self.model.eval()

        # Tracking variables
        predictions, true_labels = [], []

        # Predict
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        print('    DONE.')


class BERTbaseFineTunedNextTimeTextEncoder(TextEncoder):

    def __init__(self, language="english", encoding_length=20, max_sentence_length_buffer=12, batch_size=32,
                 lr=2e-5, eps=1e-8, epochs=2, hidden_states_representations='cls', text_base_for_training='event',
                 model_dir="./models"):
        self.name = "BERTbaseFineTunedNextTime"
        self.model = None
        self.tokenizer = None
        self.max_sentence_length = 0
        self.max_sentence_length_buffer = max_sentence_length_buffer
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.epochs = epochs
        self.hidden_states_representations = hidden_states_representations
        self.text_base_for_training = text_base_for_training
        self.name = f"{self.name}_{text_base_for_training}_{hidden_states_representations}_e{epochs}_lr{lr}"
        self.model_dir = model_dir
        self.device = None
        super().__init__(language=language, encoding_length=encoding_length)

    def fit(self, docs, labels):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

        self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=1,  # Regression; then, loss is MSELoss()
                output_attentions=False,  # Return attention weights
                output_hidden_states=True  # Return all hidden states
        )
        self.print_model_specifications()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Determine max. sentence length (for padding / truncating)
        for doc in docs:
            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = self.tokenizer.encode(doc, add_special_tokens=True)
            # Update the maximum sentence length.
            self.max_sentence_length = max(self.max_sentence_length, len(input_ids))

        if self.max_sentence_length > 512:
            self.max_sentence_length = 512
        # Add buffer in case there are some longer test sentences
        self.max_sentence_length += self.max_sentence_length_buffer

        docs = self.preprocess_docs(docs, as_list=False)
        input_ids, attention_masks, labels = self.bert_input_preparation(docs, labels, for_training=True)
        train_dataloader, validation_dataloader = self.get_dataloader(input_ids, attention_masks, labels)
        self.bert_fine_tuning(train_dataloader, validation_dataloader)

        return self

    def transform(self, docs):
        docs = self.preprocess_docs(docs, as_list=False)
        input_ids, attention_masks, _ = self.bert_input_preparation(docs, labels=None, for_training=False)
        embeddings = self.get_bert_embeddings(input_ids, attention_masks,
                                              hidden_states_representations=self.hidden_states_representations)
        return embeddings

    def bert_input_preparation(self, docs, labels=None, for_training=True):
        """Preparing the input for BERT"""

        input_ids = []
        attention_masks = []

        for doc in docs:
            encoded_dict = self.tokenizer.encode_plus(
                    doc,
                    add_special_tokens=True,  # Add [CLS] and [SEP]
                    max_length=self.max_sentence_length,  # Pad or truncate sentence
                    # pad_to_max_length=True,
                    # truncation=True,
                    padding='max_length',
                    return_attention_mask=True,  # Construct attention masks
                    return_tensors='pt'  # Return pytorch tensors
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        if for_training:
            labels = torch.tensor(labels, dtype=torch.long)

        return input_ids, attention_masks, labels

    def get_dataloader(self, input_ids, attention_masks, labels=None, for_training=True):

        if for_training:
            # 90-10 Train-validation split
            dataset = TensorDataset(input_ids, attention_masks, labels)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size

            # Divide dataset by randomly selecting samples
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            print('{:>5,} training samples'.format(train_size))
            print('{:>5,} validation samples'.format(val_size))

            # Take training samples in random order
            train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=self.batch_size)
            # For validation the order doesn't matter, so just take them sequentially
            validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                               batch_size=self.batch_size)
            return train_dataloader, validation_dataloader

        else:
            prediction_data = TensorDataset(input_ids, attention_masks, labels)
            prediction_sampler = SequentialSampler(prediction_data)
            prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.batch_size)
            return prediction_dataloader

    def bert_fine_tuning(self, train_dataloader, validation_dataloader):

        print("")
        print("Fine-tuning BERT...")

        model_path = f"{self.model_dir}/{self.name}"
        if pathlib.Path(model_path).exists():
            # below for testing purposes: simply load previously fine-tuned model
            self.model = torch.load(model_path)
            print("Existing fine-tuned model loaded.")
            return

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # 'W' stands for 'Weight Decay fix" ??
        optimizer = transformers.AdamW(self.model.parameters(), lr=self.lr, eps=self.eps)
        # Total number of training steps is [number of batches] x [number of epochs]
        total_steps = len(train_dataloader) * self.epochs
        # Create learning rate scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                                 num_training_steps=total_steps)

        # # Set seed for reproducible results
        # seed_val = 0
        # random.seed(seed_val)
        # np.random.seed(seed_val)
        # torch.manual_seed(seed_val)
        # torch.cuda.manual_seed_all(seed_val)

        training_stats = []
        total_t0 = time.time()

        for epoch_i in range(0, self.epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 20 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # are given and what flags are set. For our usage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                print(b_input_ids.size(), b_input_mask.size(), b_labels.size())
                loss, logits, _ = self.model(b_input_ids,
                                             token_type_ids=None,
                                             attention_mask=b_input_mask,
                                             labels=torch.tensor(b_labels, dtype=torch.float),
                                             return_dict=False)

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
            self.model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    (loss, logits, _) = self.model(b_input_ids,
                                                   token_type_ids=None,
                                                   attention_mask=b_input_mask,
                                                   labels=b_labels,
                                                   return_dict=False)

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and accumulate it over all batches.
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                    {
                        'epoch': epoch_i + 1,
                        'Training Loss': avg_train_loss,
                        'Valid. Loss': avg_val_loss,
                        'Valid. Accur.': avg_val_accuracy,
                        'Training Time': training_time,
                        'Validation Time': validation_time
                    }
            )

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # save fine-tuned model
        torch.save(self.model, model_path)

        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time() - total_t0)))
        self.print_training_summary(training_stats)

    def get_bert_embeddings(self, input_ids, attention_masks, hidden_states_representations='cls'):

        # Gradient calculation id disabled, model is in inference mode
        with torch.no_grad():
            # BertForSequenceClassification is a wrapper that consists of two parts:
            # (1) BERT model (attribute 'bert') and (2) a classifier (attribute 'classifier').
            # here, call the BERT model directly to get the hidden states of the model
            outputs = self.model.bert(input_ids, attention_masks)
            last_hidden_states = outputs[0]
            cls_hidden_state = last_hidden_states[:, 0, :]
            hidden_states = outputs[2]

        if hidden_states_representations == "cls":
            embeddings = cls_hidden_state.numpy()
        if hidden_states_representations == "last_4_mean":
            embeddings = torch.mean(torch.stack(hidden_states[-4:], dim=0)[:, :, 0, :], dim=0).numpy()
        elif hidden_states_representations == "last_4_sum":
            sum_last_4_layers = torch.stack(hidden_states[-4:]).sum(0)
            embeddings = sum_last_4_layers.numpy()
        elif hidden_states_representations == "last_4_cat":
            concat_last_4_layers = torch.stack(hidden_states[-4:], dim=0)
            embeddings = concat_last_4_layers.numpy()

        return embeddings

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(self, elapsed):
        # Returns elapsed time, takes a time in seconds and returns a string hh:mm:ss
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def print_model_specifications(self):
        # Get all of the model's parameters as a list of tuples.
        params = list(self.model.named_parameters())

        print('The BERT model has {:} different named parameters.\n'.format(len(params)))

        print('==== Embedding Layer ====\n')

        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== First Transformer ====\n')

        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')

        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    def print_training_summary(self, training_stats):

        # Display floats with two decimal places.
        pd.set_option('precision', 2)

        # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=training_stats)

        # Use the 'epoch' as the row index.
        df_stats = df_stats.set_index('epoch')

        # A hack to force the column headers to wrap.
        # df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

        # Display the table
        df_stats

        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        # Plot the learning curve.
        plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
        plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

        # Label the plot.
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks([1, 2, 3, 4])

        plt.show()

    def evaluate_finetuned_model(self, prediction_dataloader):

        # Prediction on test set

        print('Predicting labels for {:,} test sentences...'.format(len(prediction_dataloader)))

        # Put model in evaluation mode
        self.model.eval()

        # Tracking variables
        predictions, true_labels = [], []

        # Predict
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        print('    DONE.')


class BERTbaseFineTunedNextActivityAndTimeTextEncoder(TextEncoder):

    def __init__(self, language="english", encoding_length=20, max_sentence_length_buffer=12, batch_size=32,
                 lr=2e-5, eps=1e-8, epochs=2, hidden_states_representations='cls', text_base_for_training='event',
                 model_dir="./models"):
        self.name = "BERTbaseFineTunedNextActivityAndTime"
        self.name_act = "BERTbaseFineTunedNextActivity"
        self.name_time = "BERTbaseFineTunedNextTime"
        self.model_act = None
        self.model_time = None
        self.tokenizer = None
        self.max_sentence_length = 0
        self.max_sentence_length_buffer = max_sentence_length_buffer
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.epochs = epochs
        self.hidden_states_representations = hidden_states_representations
        self.text_base_for_training = text_base_for_training
        self.name = f"{self.name}_{text_base_for_training}_{hidden_states_representations}_e{epochs}_lr{lr}"
        self.name_act = f"{self.name_act}_{text_base_for_training}_{hidden_states_representations}_e{epochs}_lr{lr}"
        self.name_time = f"{self.name_time}_{text_base_for_training}_{hidden_states_representations}_e{epochs}"
        self.model_dir = model_dir
        self.device = None
        super().__init__(language=language, encoding_length=encoding_length)

    def fit(self, docs, labels):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

        model_path_act = f"{self.model_dir}/{self.name_act}"
        if pathlib.Path(model_path_act).exists():
            # below for testing purposes: simply load previously fine-tuned model
            self.model_act = torch.load(model_path_act)
            print("Existing fine-tuned model (activity) loaded.")

        model_path_time = f"{self.model_dir}/{self.name_time}"
        if pathlib.Path(model_path_time).exists():
            # below for testing purposes: simply load previously fine-tuned model
            self.model_time = torch.load(model_path_time)
            print("Existing fine-tuned model (time) loaded.")

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Determine max. sentence length (for padding / truncating)
        for doc in docs:
            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = self.tokenizer.encode(doc, add_special_tokens=True)
            # Update the maximum sentence length.
            self.max_sentence_length = max(self.max_sentence_length, len(input_ids))

        if self.max_sentence_length > 512:
            self.max_sentence_length = 512
        # Add buffer in case there are some longer test sentences
        self.max_sentence_length += self.max_sentence_length_buffer
        #
        # docs = self.preprocess_docs(docs, as_list=False)
        # input_ids, attention_masks, labels = self.bert_input_preparation(docs, labels, for_training=True)
        # train_dataloader, validation_dataloader = self.get_dataloader(input_ids, attention_masks, labels)
        # self.bert_fine_tuning(train_dataloader, validation_dataloader)

        return self

    def transform(self, docs):
        docs = self.preprocess_docs(docs, as_list=False)
        input_ids, attention_masks, _ = self.bert_input_preparation(docs, labels=None, for_training=False)
        embeddings_act, embeddings_time = self.get_bert_embeddings(input_ids, attention_masks,
                                                                   hidden_states_representations=self.hidden_states_representations)
        return embeddings_act, embeddings_time

    def bert_input_preparation(self, docs, labels=None, for_training=True):
        """Preparing the input for BERT"""

        input_ids = []
        attention_masks = []

        for doc in docs:
            encoded_dict = self.tokenizer.encode_plus(
                    doc,
                    add_special_tokens=True,  # Add [CLS] and [SEP]
                    max_length=self.max_sentence_length,  # Pad or truncate sentence
                    # pad_to_max_length=True,
                    # truncation=True,
                    padding='max_length',
                    return_attention_mask=True,  # Construct attention masks
                    return_tensors='pt'  # Return pytorch tensors
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        if for_training:
            labels = torch.tensor(labels, dtype=torch.long)

        return input_ids, attention_masks, labels

    def get_bert_embeddings(self, input_ids, attention_masks, hidden_states_representations='cls'):

        # Gradient calculation id disabled, model is in inference mode
        with torch.no_grad():
            # BertForSequenceClassification is a wrapper that consists of two parts:
            # (1) BERT model (attribute 'bert') and (2) a classifier (attribute 'classifier').
            # here, call the BERT model directly to get the hidden states of the model

            # BERT fine-tuned toward next activity prediction
            outputs_act = self.model_act.bert(input_ids, attention_masks)
            last_hidden_states_act = outputs_act[0]
            cls_hidden_state_act = last_hidden_states_act[:, 0, :]
            hidden_states_act = outputs_act[2]

            # BERT fine-tuned toward next activity prediction
            outputs_time = self.model_time.bert(input_ids, attention_masks)
            last_hidden_states_time = outputs_time[0]
            cls_hidden_state_time = last_hidden_states_time[:, 0, :]
            hidden_states_time = outputs_time[2]

        if hidden_states_representations == "cls":
            embeddings_act = cls_hidden_state_act.numpy()
            embeddings_time = cls_hidden_state_time.numpy()
        if hidden_states_representations == "last_4_mean":
            embeddings_act = torch.mean(torch.stack(hidden_states_act[-4:], dim=0)[:, :, 0, :], dim=0).numpy()
            embeddings_time = torch.mean(torch.stack(hidden_states_time[-4:], dim=0)[:, :, 0, :], dim=0).numpy()
        elif hidden_states_representations == "last_4_sum":
            sum_last_4_layers_act = torch.stack(hidden_states_act[-4:]).sum(0)
            sum_last_4_layers_time = torch.stack(hidden_states_time[-4:]).sum(0)
            embeddings_act = sum_last_4_layers_act.numpy()
            embeddings_time = sum_last_4_layers_time.numpy()
        elif hidden_states_representations == "last_4_cat":
            concat_last_4_layers_act = torch.stack(hidden_states_act[-4:], dim=0)
            concat_last_4_layers_time = torch.stack(hidden_states_time[-4:], dim=0)
            embeddings_act = concat_last_4_layers_act.numpy()
            embeddings_time = concat_last_4_layers_time.numpy()

        return embeddings_act, embeddings_time

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(self, elapsed):
        # Returns elapsed time, takes a time in seconds and returns a string hh:mm:ss
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def print_model_specifications(self):
        # Get all of the model's parameters as a list of tuples.
        params = list(self.model.named_parameters())

        print('The BERT model has {:} different named parameters.\n'.format(len(params)))

        print('==== Embedding Layer ====\n')

        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== First Transformer ====\n')

        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')

        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    def print_training_summary(self, training_stats):

        # Display floats with two decimal places.
        pd.set_option('precision', 2)

        # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=training_stats)

        # Use the 'epoch' as the row index.
        df_stats = df_stats.set_index('epoch')

        # A hack to force the column headers to wrap.
        # df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

        # Display the table
        df_stats

        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        # Plot the learning curve.
        plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
        plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

        # Label the plot.
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks([1, 2, 3, 4])

        plt.show()

    def evaluate_finetuned_model(self, prediction_dataloader):

        # Prediction on test set

        print('Predicting labels for {:,} test sentences...'.format(len(prediction_dataloader)))

        # Put model in evaluation mode
        self.model.eval()

        # Tracking variables
        predictions, true_labels = [], []

        # Predict
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        print('    DONE.')


class BERTfromScratchTextEncoder(TextEncoder):
    """ This version of BERT is trained from scratch. The tokenizer is the pre-trained version from the 'transformers'
    library from huggingface. """

    def __init__(self, language="english", encoding_length=36, max_sentence_length_buffer=12,
                 text_doc_path="../datasets/questions.txt", model_dir="../models"):
        self.name = f"BERTfromScratch_s{encoding_length}"
        self.model = None
        self.tokenizer = None
        self.max_sentence_length = 0
        self.max_sentence_length_buffer = max_sentence_length_buffer
        self.encoding_length = encoding_length
        self.text_doc_path = text_doc_path
        self.bert_dir = f"{model_dir}/{self.name}/bert/"
        if not os.path.exists(self.bert_dir):
            os.makedirs(self.bert_dir)

        super().__init__(language=language, encoding_length=encoding_length)

    def fit(self, docs, **kwargs):
        configuration = BertConfig(hidden_size=self.encoding_length, intermediate_size=4 * self.encoding_length)
        self.model = BertForPreTraining(configuration)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        pretrain_bert_from_scratch(self.model, self.tokenizer, self.encoding_length, dataset_path=self.text_doc_path, out_dir=self.bert_dir)
        self.model = BertModel.from_pretrained(self.bert_dir)
        return self

    def transform(self, docs):
        docs = self.preprocess_docs(docs, as_list=False)
        list_embeddings = []
        for doc in docs:
            tokenized_text, tokens_tensor, segments_tensor = self.bert_text_preparation(doc)
            list_embeddings.append(self.get_bert_embeddings(tokens_tensor, segments_tensor, embedding_level="cls"))
        return np.squeeze(np.array(list_embeddings), axis=0)

    def bert_text_preparation(self, text):
        """Preparing the input for BERT

        Takes a string argument and performs pre-processing like adding special tokens, tokenization, tokens to ids,
        and tokens to segment ids. All tokens are mapped to segment id = 1.

        Args:
            text (str): Text to be converted
            tokenizer (obj): Tokenizer object to convert text into BERT-readable tokens and ids

        Returns:
            list: List of BERT-readable tokens
            obj: Torch tensor with token ids
            obj: Torch tensor segment ids

        """
        # 1. Add special tokens to start and end of sentence
        # [SEP] token at the end of each sentence
        # [CLS] token at beginning of each sentence; the final hidden state corresponding to this token is used as the
        # aggregate sequence representation for classification tasks
        marked_text = "[CLS] " + text + " [SEP]"  # here, [CLS] is ignored
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokenized_text, tokens_tensor, segments_tensors

    def get_bert_embeddings(self, tokens_tensor, segments_tensors, embedding_level="cls"):
        """Get embeddings from an embedding model

        Args:
            tokens_tensor (obj): Torch tensor size [n_tokens] with token ids for each token in text
            segments_tensors (obj): Torch tensor size [n_tokens] with segment ids for each token in text
            model (obj): Embedding model to generate embeddings from token and segment ids

        Returns:
            obj: Numpy array of floats of size [n_tokens, n_embedding_dimensions] containing embeddings for each token

        """

        # Gradient calculation id disabled, model is in inference mode
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            # 'hidden states' is a tensor with shape [layers, batch, token, feature/hidden unit]
            # layers is 13 (= 12 BERT layers + 1 input embeddings)
            last_hidden_state = outputs[0]

        if embedding_level == "cls":
            # # [CLS] embeddings are the aggregate sequence representation for classification tasks
            final_embeddings = last_hidden_state[:, 0, :]
            return final_embeddings.numpy()


class BERTAndTokenizerFromScratchTextEncoder(TextEncoder):
    """ This version of BERT is trained from scratch. Additionally, the tokenizer is trained from scratch as well. """

    def __init__(self, language="english", encoding_length=36, max_sentence_length_buffer=12, vocab_size=1000,
                 text_doc_path="../datasets/questions.txt", model_dir="../models"):
        self.name = f"BERTAndTokenizerFromScratch_s{encoding_length}_v{vocab_size}"
        self.model = None
        self.tokenizer = None
        self.max_sentence_length = 0
        self.max_sentence_length_buffer = max_sentence_length_buffer
        self.encoding_length = encoding_length
        self.text_doc_path = text_doc_path
        self.vocab_size = vocab_size
        self.bert_dir = f"{model_dir}/{self.name}/bert/"
        self.tokenizer_dir = f"{model_dir}/{self.name}/tokenizer/"
        super().__init__(language=language, encoding_length=encoding_length)

    def fit(self, docs, **kwargs):
        configuration = BertConfig(hidden_size=self.encoding_length, intermediate_size=4 * self.encoding_length)
        if not os.path.exists(self.tokenizer_dir):
            os.makedirs(self.tokenizer_dir)
            train_tokenizer_from_scratch(vocab_size=self.vocab_size, dataset_path=self.text_doc_path, out_dir=self.tokenizer_dir)
        self.tokenizer = BertTokenizer(vocab_file=f"{self.tokenizer_dir}vocab.txt")
        if not os.path.exists(self.bert_dir):
            os.makedirs(self.bert_dir)
            self.model = BertForPreTraining(configuration)
            pretrain_bert_from_scratch(self.model, self.tokenizer, self.encoding_length, dataset_path=self.text_doc_path, out_dir=self.bert_dir)
        self.model = BertModel.from_pretrained(self.bert_dir)
        return self

    def transform(self, docs):
        docs = self.preprocess_docs(docs, as_list=False)
        list_embeddings = []
        for doc in docs:
            tokenized_text, tokens_tensor, segments_tensor = self.bert_text_preparation(doc)
            list_embeddings.append(self.get_bert_embeddings(tokens_tensor, segments_tensor, embedding_level="cls"))
        return np.squeeze(np.array(list_embeddings), axis=0)

    def bert_text_preparation(self, text):
        """Preparing the input for BERT

        Takes a string argument and performs pre-processing like adding special tokens, tokenization, tokens to ids,
        and tokens to segment ids. All tokens are mapped to segment id = 1.

        Args:
            text (str): Text to be converted
            tokenizer (obj): Tokenizer object to convert text into BERT-readable tokens and ids

        Returns:
            list: List of BERT-readable tokens
            obj: Torch tensor with token ids
            obj: Torch tensor segment ids

        """
        # 1. Add special tokens to start and end of sentence
        # [SEP] token at the end of each sentence
        # [CLS] token at beginning of each sentence; the final hidden state corresponding to this token is used as the
        # aggregate sequence representation for classification tasks
        marked_text = "[CLS] " + text + " [SEP]"  # here, [CLS] is ignored
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokenized_text, tokens_tensor, segments_tensors

    def get_bert_embeddings(self, tokens_tensor, segments_tensors, embedding_level="cls"):
        """Get embeddings from an embedding model

        Args:
            tokens_tensor (obj): Torch tensor size [n_tokens] with token ids for each token in text
            segments_tensors (obj): Torch tensor size [n_tokens] with segment ids for each token in text
            model (obj): Embedding model to generate embeddings from token and segment ids

        Returns:
            obj: Numpy array of floats of size [n_tokens, n_embedding_dimensions] containing embeddings for each token

        """

        # Gradient calculation id disabled, model is in inference mode
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            # 'hidden states' is a tensor with shape [layers, batch, token, feature/hidden unit]
            # layers is 13 (= 12 BERT layers + 1 input embeddings)
            last_hidden_state = outputs[0]

        if embedding_level == "cls":
            # # [CLS] embeddings are the aggregate sequence representation for classification tasks
            final_embeddings = last_hidden_state[:, 0, :]
            return final_embeddings.numpy()


def pretrain_bert_from_scratch(bert_model, bert_tokenizer, encoding_size, dataset_path, out_dir):

    training_args = TrainingArguments(output_dir=out_dir,
                                      overwrite_output_dir=True,
                                      num_train_epochs=3,  # default=3.0
                                      per_device_train_batch_size=32,
                                      save_steps=1000,
                                      save_total_limit=3,
                                      )

    data_collator = DataCollatorForLanguageModeling(tokenizer=bert_tokenizer,
                                                    mlm=True,
                                                    mlm_probability=0.15)

    train_dataset = TextDatasetForNextSentencePrediction(tokenizer=bert_tokenizer,
                                                         file_path=dataset_path,
                                                         block_size=512  # 256
                                                         )

    trainer = Trainer(model=bert_model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_dataset,
                      tokenizer=bert_tokenizer)

    trainer.train()
    trainer.save_model(out_dir)


def train_tokenizer_from_scratch(vocab_size, dataset_path, out_dir):
    # Initialize an empty tokenizer
    tokenizer = BertWordPieceTokenizer(clean_text=True,
                                       handle_chinese_chars=True,
                                       strip_accents=True,
                                       lowercase=True)

    # And then train
    tokenizer.train(dataset_path,
                    vocab_size=vocab_size,
                    min_frequency=2,
                    show_progress=True,
                    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                    limit_alphabet=1000,
                    wordpieces_prefix="##",
                    )

    # Save the files
    tokenizer.save_model(out_dir)
