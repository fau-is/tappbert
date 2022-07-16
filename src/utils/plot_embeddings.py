from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import torch
import pandas as pd
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from gensim.models import Doc2Vec
from matplotlib.backends.backend_pdf import PdfPages
import os

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

sns.set(style='dark')

language = "english"
encoding_length = 768
encoding_length = encoding_length
stop_words = set(stopwords.words(language))
stop_words.remove("not")
stop_words.remove("no")
lemmatizer = WordNetLemmatizer() if language == "english" else None
stemmer = SnowballStemmer(language)


def get_visual_embs_from_bert(sentence):
    """Get BERT embedding for the sentence,
    project it to a 2D subspace where [CLS] is (1,0) and [SEP] is (0,1)."""
    embs = bert_embedding([sentence], filter_spec_tokens=False)
    tokens = embs[0]
    print(tokens)
    embV = embs[1]
    W = np.array(embV)
    B = np.array([embV[0], embV[-1]])
    Bi = np.linalg.pinv(B.T)
    Wp = np.matmul(Bi, W.T)
    print(Wp)
    return Wp, tokens


def bert_embedding(sentence, filter_spec_tokens=False):
    tokenized_text, tokens_tensor, segments_tensor = bert_text_preparation(sentence[0])
    embeddings = get_bert_embeddings(tokens_tensor, segments_tensor)
    return (tokenized_text, embeddings)


def bert_text_preparation(text):
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
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors):
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
        # BertForSequenceClassification is a wrapper that consists of two parts:
        # (1) BERT model (attribute 'bert') and (2) a classifier (attribute 'classifier').
        # here, call the BERT model directly to get the hidden states of the model
        outputs = model.bert(tokens_tensor, segments_tensors)
        last_hidden_state = outputs[0]
        cls_hidden_state = last_hidden_state[0, :, :]
        embedding = cls_hidden_state.numpy()

        return embedding


def preprocess_docs(docs, as_list=True):
    docs_preprocessed = []
    for doc in docs:
        doc = doc.lower()
        words = word_tokenize(doc, language="english")
        if lemmatizer is not None:
            words_processed = [lemmatizer.lemmatize(word) for word in words if
                               word not in stop_words and word.isalpha()]
        else:
            words_processed = [stemmer.stem(word) for word in words if
                               word not in stop_words and word.isalpha()]
        if as_list:
            docs_preprocessed.append(words_processed)
        else:
            docs_preprocessed.append(" ".join(words_processed))
    return docs_preprocessed


def get_visual_embs_from_doc2vec(doc2vec, sentence):
    w = preprocess_docs([sentence])
    wv = []
    for i in range(len(w[0])):
        # if w[0][i] == "right":
        #     print(doc2vec[w[0][i]], doc2vec.infer_vector([w[0][i]])[:3])
        doc2vec.random.seed(0)
        wv.append(doc2vec.infer_vector([w[0][i]]))
        # wv.append(doc2vec[w[0][i]])
    W = np.array(wv)

    B = np.array([W.T[:, 0], W.T[:, -1]])
    Bi = np.linalg.pinv(B.T)
    Wp = np.matmul(Bi, W.T)

    return Wp, w[0]


def plot_sentence_embeddings_bert(s, e, t, ambiguous_word, c, text_model, plot_all_tokens=True):
    plt.figure(figsize=(20, 15))

    colors = []
    for color in c:
        if color == "blue":
            colors.append({"marker": "navy", "arrow": "cornflowerblue"})
        elif color == "red":
            colors.append({"marker": "red", "arrow": "lightcoral"})
        elif color == "green":
            colors.append({"marker": "forestgreen", "arrow": "darkseagreen"})
        elif color == "yellow":
            colors.append({"marker": "darkorange", "arrow": "sandybrown"})

    color_palette = {
        f"{s[0]}": colors[0]["marker"],
        f"{s[1]}": colors[1]["marker"],
        f"{s[2]}": colors[2]["marker"],
    }

    df0 = pd.DataFrame({"x": e[0][0], "y": e[0][1], "sentence": s[0], "t": t[0]})
    df1 = pd.DataFrame({"x": e[1][0], "y": e[1][1], "sentence": s[1], "t": t[1]})
    df2 = pd.DataFrame({"x": e[2][0], "y": e[2][1], "sentence": s[2], "t": t[2]})

    df = pd.concat([df0, df1, df2])

    p = sns.scatterplot(x=df["x"], y=df["y"], hue=df["sentence"], palette=color_palette)

    X = df["x"].values
    Y = df["y"].values
    H = df["sentence"].values
    T = df["t"].values

    counter_color = 0
    color = colors[counter_color]
    for i, (x, y, h, t) in enumerate(zip(X, Y, H, T)):
        if i < len(H) - 1:
            if h != H[i + 1]:
                # switch color for new sentence
                counter_color += 1
                color = colors[counter_color]
            else:
                # draw arrow to next word, except for last word
                p.arrow(x, y, X[i + 1] - x, Y[i + 1] - y,
                        shape='full', color=color["arrow"], length_includes_head=True, head_width=0.005, lw=0.2,
                        # lw=d['MAG'] / 2., , head_length=3.,
                        )
        if plot_all_tokens:
            if t == ambiguous_word:
                txt = p.text(x + 0.0075, y, t,
                             horizontalalignment='left', fontsize=15, color=color["arrow"], weight='heavy', zorder=10)
                if c[counter_color] == "blue":
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="mediumblue")])
                elif c[counter_color] == "red":
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="darkred")])
                elif c[counter_color] == "green":
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="darkgreen")])
            else:
                txt = p.text(x + 0.0075, y, t,
                             horizontalalignment='left', fontsize=15, color="white", weight='semibold')
                txt.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground=color["marker"])])
                pass
        else:
            if t == ambiguous_word:
                txt = p.text(x + 0.0075, y, t,
                             horizontalalignment='left', fontsize=15, color=color["marker"], weight='semibold', zorder=15)
            else:
                pass

    plt.xticks(np.arange(0, 1.1, step=0.2), fontsize=15)
    plt.yticks(np.arange(0, 1.1, step=0.2), fontsize=15)
    plt.xlabel("X normalized by [CLS]", fontsize=15, labelpad=15)
    plt.ylabel("Y normalized by [SEP]", fontsize=15, labelpad=15)

    p.legend(fontsize=15).set_title(None)
    p.grid(linewidth=0.5)
    plt.savefig(f"embeddings_right_{text_model}.png")
    plt.show()


def plot_sentence_embeddings_d2v(s, e, t, ambiguous_word, c, text_model, plot_all_tokens):
    plt.figure(figsize=(20, 15))

    colors = []
    for color in c:
        if color == "blue":
            colors.append({"marker": "navy", "arrow": "cornflowerblue"})
        elif color == "red":
            colors.append({"marker": "red", "arrow": "lightcoral"})
        elif color == "green":
            colors.append({"marker": "forestgreen", "arrow": "darkseagreen"})
        elif color == "yellow":
            colors.append({"marker": "darkorange", "arrow": "sandybrown"})

    color_palette = {
        f"{s[0]}": colors[0]["marker"],
        f"{s[1]}": colors[1]["marker"],
        f"{s[2]}": colors[2]["marker"],
    }

    df0 = pd.DataFrame({"x": e[0][0], "y": e[0][1], "sentence": s[0], "t": t[0]})
    df1 = pd.DataFrame({"x": e[1][0], "y": e[1][1], "sentence": s[1], "t": t[1]})
    df2 = pd.DataFrame({"x": e[2][0], "y": e[2][1], "sentence": s[2], "t": t[2]})

    df = pd.concat([df0, df1, df2])

    p = sns.scatterplot(x=df["x"], y=df["y"], hue=df["sentence"], palette=color_palette)

    X = df["x"].values
    Y = df["y"].values
    H = df["sentence"].values
    T = df["t"].values

    counter_color = 0
    color = colors[counter_color]
    offset_y = 0.025
    for i, (x, y, h, t) in enumerate(zip(X, Y, H, T)):

        if plot_all_tokens:
            if t == ambiguous_word:
                txt = p.text(x + 0.0075, y + offset_y, t,
                             horizontalalignment='left', fontsize=15, color=color["arrow"], weight='heavy', zorder=10)
                if c[counter_color] == "blue":
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="mediumblue")])
                elif c[counter_color] == "red":
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="darkred")])
                elif c[counter_color] == "green":
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="darkgreen")])

            else:
                txt = p.text(x + 0.0075, y + offset_y, t,
                             horizontalalignment='left', fontsize=15, color="white", weight='semibold')
                txt.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground=color["marker"])])
                pass
        else:
            if t == ambiguous_word:
                txt = p.text(x + 0.0075, y + offset_y, t,
                             horizontalalignment='left', fontsize=15, color=color["marker"], weight='semibold', zorder=15)
            else:
                pass

        if i < len(H) - 1:
            if h == H[i + 1]:
                # draw arrow to next word, except for last word
                p.arrow(x, y, X[i + 1] - x, Y[i + 1] - y,
                        shape='full', color=color["arrow"], length_includes_head=True, head_width=0.005, lw=0.2,
                        # lw=d['MAG'] / 2., , head_length=3.,
                        )
            else:
                # switch color for new sentence
                counter_color += 1
                color = colors[counter_color]
                if counter_color % 2 == 0:
                    offset_y = 0
                else:
                    offset_y = -0.025

    plt.xticks(np.arange(0, 1.1, step=0.2), fontsize=15)
    plt.yticks(np.arange(0, 1.1, step=0.2), fontsize=15)
    plt.xlabel("X", fontsize=15, labelpad=15)
    plt.ylabel("Y", fontsize=15, labelpad=15)

    p.legend(fontsize=15).set_title(None)
    p.grid(linewidth=0.5)
    plt.savefig(f"embeddings_right_{text_model}.png")
    plt.show()


def plot_sentence_embeddings_bert_subplot(s, e, t, ambiguous_word, c, plot_all_tokens, axs, x_pos, y_pos):
    fontsize = 24

    colors = []
    for color in c:
        if color == "blue":
            colors.append({"marker": "blue", "arrow": "cornflowerblue"})
        elif color == "red":
            colors.append({"marker": "darkorchid", "arrow": "plum"})
        elif color == "green":
            colors.append({"marker": "forestgreen", "arrow": "darkseagreen"})
        elif color == "yellow":
            colors.append({"marker": "chocolate", "arrow": "sandybrown"})
        elif color == "black":
            colors.append({"marker": "black", "arrow": "grey"})

    color_palette = {
        f"{s[0]}": colors[0]["marker"],
        f"{s[1]}": colors[1]["marker"],
        f"{s[2]}": colors[2]["marker"],
    }

    df0 = pd.DataFrame({"x": e[0][0], "y": e[0][1], "sentence": s[0], "t": t[0]})
    df1 = pd.DataFrame({"x": e[1][0], "y": e[1][1], "sentence": s[1], "t": t[1]})
    df2 = pd.DataFrame({"x": e[2][0], "y": e[2][1], "sentence": s[2], "t": t[2]})

    df = pd.concat([df0, df1, df2])

    sns.scatterplot(x=df["x"], y=df["y"], hue=df["sentence"], palette=color_palette, ax=axs[x_pos],
                    style=df["sentence"],
                    markers=["o", "v", "P"], edgecolor='none', legend=None)

    X = df["x"].values
    Y = df["y"].values
    H = df["sentence"].values
    T = df["t"].values

    counter_color = 0
    color = colors[counter_color]
    for i, (x, y, h, t) in enumerate(zip(X, Y, H, T)):
        if i < len(H) - 1:
            if h != H[i + 1]:
                # switch color for new sentence
                counter_color += 1
                color = colors[counter_color]
            else:
                # draw arrow to next word, except for last word
                axs[x_pos].arrow(x, y, X[i + 1] - x, Y[i + 1] - y,
                                 shape='full',
                                 color=color["arrow"],
                                 length_includes_head=True,
                                 head_width=0.005,
                                 lw=0.2)
        if plot_all_tokens:
            if t == ambiguous_word:
                txt = axs[x_pos].text(x + 0.05, y + 0.01, t,
                                      horizontalalignment='center',
                                      fontsize=fontsize,
                                      color=color["marker"],
                                      weight='heavy',
                                      zorder=10)
            else:
                txt = axs[x_pos].text(x + 0.05, y + 0.01, t,
                                      horizontalalignment='center',
                                      fontsize=fontsize,
                                      color=color["arrow"],
                                      weight='normal')
                pass
        else:
            if t == ambiguous_word:
                txt = axs[x_pos].text(x + 0.05, y + 0.01, t,
                                      horizontalalignment='center',
                                      fontsize=fontsize,
                                      color=color["marker"],
                                      weight='semibold',
                                      zorder=15)
            else:
                pass

    axs[x_pos].set_xlim([-0.2, 1.2])
    axs[x_pos].set_ylim([-0.2, 1.2])
    axs[x_pos].set_xticks(np.arange(-0.2, 1.3, step=0.2))
    axs[x_pos].set_yticks(np.arange(-0.2, 1.3, step=0.2))

    axs[x_pos].set_xlabel("x", fontsize=fontsize)
    axs[x_pos].set_ylabel("y", fontsize=fontsize)
    axs[x_pos].tick_params(axis='both', which='major', labelsize=fontsize)

    axs[x_pos].grid(linewidth=0.5)


def plot_sentence_embeddings_d2v_subplot(s, e, t, ambiguous_word, c, text_model, plot_all_tokens, axs, x_pos, y_pos):
    fontsize = 24

    colors = []
    for color in c:
        if color == "blue":
            colors.append({"marker": "blue", "arrow": "cornflowerblue"})
        elif color == "red":
            colors.append({"marker": "darkorchid", "arrow": "plum"})
        elif color == "green":
            colors.append({"marker": "forestgreen", "arrow": "darkseagreen"})
        elif color == "yellow":
            colors.append({"marker": "chocolate", "arrow": "sandybrown"})
        elif color == "black":
            colors.append({"marker": "black", "arrow": "grey"})

    color_palette = {
        f"{s[0]}": colors[0]["marker"],
        f"{s[1]}": colors[1]["marker"],
        f"{s[2]}": colors[2]["marker"],
    }

    df0 = pd.DataFrame({"x": e[0][0], "y": e[0][1], "sentence": s[0], "t": t[0]})
    df1 = pd.DataFrame({"x": e[1][0], "y": e[1][1], "sentence": s[1], "t": t[1]})
    df2 = pd.DataFrame({"x": e[2][0], "y": e[2][1], "sentence": s[2], "t": t[2]})

    df = pd.concat([df0, df1, df2])

    sns.scatterplot(x=df["x"], y=df["y"], hue=df["sentence"], palette=color_palette, ax=axs[x_pos],
                    style=df["sentence"],
                    markers=["o", "v", "P"], edgecolor='none', )

    X = df["x"].values
    Y = df["y"].values
    H = df["sentence"].values
    T = df["t"].values

    counter_color = 0
    color = colors[counter_color]
    for i, (x, y, h, t) in enumerate(zip(X, Y, H, T)):
        # duplicate token
        unique, counts = np.unique(T[:i], return_counts=True)
        t_occur = dict(zip(unique, counts))
        if t in t_occur.keys():
            offset_y = t_occur[t] * 0.04
        else:
            offset_y = 0
        if plot_all_tokens:
            if t == ambiguous_word:
                txt = axs[x_pos].text(x + 0.05, y + 0.01 + offset_y, t,
                                      horizontalalignment='center',
                                      fontsize=fontsize,
                                      color=color["marker"],
                                      weight='heavy',
                                      zorder=10)
            else:
                txt = axs[x_pos].text(x + 0.05, y + 0.01 + offset_y, t,
                                      horizontalalignment='center',
                                      fontsize=fontsize,
                                      color=color["arrow"],
                                      weight='normal')
        else:
            if t == ambiguous_word:
                txt = axs[x_pos].text(x + 0.05, y + 0.01 + offset_y, t,
                                      horizontalalignment='center',
                                      fontsize=fontsize,
                                      color=color["marker"],
                                      weight='semibold',
                                      zorder=15)
            else:
                pass

        if i < len(H) - 1:
            if h == H[i + 1]:
                # draw arrow to next word, except for last word
                arr = axs[x_pos].arrow(x, y, X[i + 1] - x, Y[i + 1] - y,
                                       shape='full',
                                       color=color["arrow"],
                                       length_includes_head=True,
                                       head_width=0.005,
                                       lw=0.2)
            else:
                # switch color for new sentence
                counter_color += 1
                color = colors[counter_color]

    axs[x_pos].set_xlim([-0.2, 1.2])
    axs[x_pos].set_ylim([-0.2, 1.2])
    axs[x_pos].set_xticks(np.arange(-0.2, 1.3, step=0.2))
    axs[x_pos].set_yticks(np.arange(-0.2, 1.3, step=0.2))

    axs[x_pos].set_xlabel("x", fontsize=fontsize)  # , labelpad=15)
    axs[x_pos].set_ylabel("y", fontsize=fontsize)  # , labelpad=15)
    axs[x_pos].tick_params(axis='both', which='major', labelsize=fontsize)

    axs[x_pos].legend().set_visible(False)
    axs[x_pos].grid(linewidth=0.5)

    return axs[x_pos].get_legend_handles_labels()


##############
# sentences
s1_right = "Can I sign a settlement agreement without my right to unemployment benefit thus jeopardizing?"
s2_right = "How do I find the right training in the table when filling in my resume?"
s3_right = "How do I find the right function name when filling in my resume?"

# figure size / layout
fig, axs = plt.subplots(1, 2, sharey='col')  # , figsize=(40, 12))

# create subplots
subplt_positions = [[0, 0], [0, 1]]

##################
# BERT
##################
# load tokenizer and bert model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = torch.load("./models/BERTbaseFineTunedNextActivity_event_cls_e16_lr5e-05")

# get bert embeddings
embeddings1, tokenized_text1 = get_visual_embs_from_bert(s1_right)
embeddings2, tokenized_text2 = get_visual_embs_from_bert(s2_right)
embeddings3, tokenized_text3 = get_visual_embs_from_bert(s3_right)

# plot bert embeddings
plot_sentence_embeddings_bert_subplot(s=[s1_right, s2_right, s3_right],
                                      e=[embeddings1, embeddings2, embeddings3],
                                      t=[tokenized_text1, tokenized_text2, tokenized_text3],
                                      ambiguous_word="right",
                                      c=["blue", "yellow", "black"],
                                      plot_all_tokens=True,
                                      axs=axs,
                                      x_pos=subplt_positions[0][1],
                                      y_pos=subplt_positions[0][1])

##################
# Doc2Vec
##################
# load doc2vec model
doc2vec = Doc2Vec.load(f"pv_100_model")

# get doc2vec embeddings
embeddings, tokenized_text = get_visual_embs_from_doc2vec(doc2vec, s1_right + s2_right + s3_right)

# plot doc2vec embeddings
handles, labels = plot_sentence_embeddings_d2v_subplot(s=[s1_right, s2_right, s3_right],
                                                       e=[embeddings[:, :9], embeddings[:, 9:15], embeddings[:, 15:]],
                                                       t=[tokenized_text[:9], tokenized_text[9:15], tokenized_text[15:]],
                                                       ambiguous_word="right",
                                                       c=["blue", "yellow", "black"],
                                                       plot_all_tokens=True,
                                                       text_model="doc2vec",
                                                       axs=axs,
                                                       x_pos=subplt_positions[1][1],
                                                       y_pos=subplt_positions[1][1])

fig.set_size_inches(20, 11)
fig.legend(handles, labels, loc='lower center', fontsize=24, bbox_to_anchor=(0, 0, 1, 1),
           bbox_transform=plt.gcf().transFigure)
fig.tight_layout()
plt.subplots_adjust(bottom=0.24)

if not os.path.exists("../plots/"):
    os.makedirs("../plots/")

plt.savefig(f"../plots/embeddings_right.png")
plt.savefig(f"../plots/embeddings_right.svg")

plt.show()

pp = PdfPages("../plots/embeddings_right.pdf")
pp.savefig(fig)
pp.close()

print("Plotting finished.")