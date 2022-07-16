import pandas as pd
import numpy as np

def compute_avg_std_results(df, method, text_dim):
    r = df[df["text_encoding"] == method]
    r = r[r["text_dim"] == text_dim]

    accuracies = r["next_activity_acc"].to_numpy()
    f1 = r["next_activity_f1_weighted"].to_numpy()
    f1_macro = r["next_activity_f1_macro"].to_numpy()
    precisions = r["next_activity_precision_weighted"].to_numpy()
    precisions_macro = r["next_activity_precision_macro"].to_numpy()
    recalls = r["next_activity_recall_weighted"].to_numpy()
    recalls_macro = r["next_activity_recall_macro"].to_numpy()
    maes = r["next_time_mae"].to_numpy()
    mses = r["next_time_mse"].to_numpy()

    avg_accuracy = round(np.mean(accuracies), 4)
    avg_f1 = round(np.mean(f1), 4)
    avg_f1_macro = round(np.mean(f1_macro), 4)
    avg_precision = round(np.mean(precisions), 4)
    avg_precision_macro = round(np.mean(precisions_macro), 4)
    avg_recall = round(np.mean(recalls), 4)
    avg_recall_macro = round(np.mean(recalls_macro), 4)
    avg_mae = round(np.mean(maes), 4)
    avg_mse = round(np.mean(mses), 4)

    std_accuracy = round(np.std(accuracies), 4)
    std_f1 = round(np.std(f1), 4)
    std_f1_macro = round(np.std(f1_macro), 4)
    std_precision = round(np.std(precisions), 4)
    std_precision_macro = round(np.std(precisions_macro), 4)
    std_recall = round(np.std(recalls), 4)
    std_recall_macro = round(np.std(recalls_macro), 4)
    std_mae = round(np.std(maes), 4)
    std_mse = round(np.std(mses), 4)

    results = {"accuracy_avg": avg_accuracy,
               "accuracy_std": std_accuracy,
               "f1_avg": avg_f1,
               "f1_avg_macro": avg_f1_macro,
               "f1_std": std_f1,
               "f1_std_macro": std_f1_macro,
               "precision_avg": avg_precision,
               "precision_avg_macro": avg_precision_macro,
               "precision_std": std_precision,
               "precision_std_macro": std_precision_macro,
               "recall_avg": avg_recall,
               "recall_avg_macro": avg_recall_macro,
               "recall_std": std_recall,
               "recall_std_macro": std_recall_macro,
               "mae_avg": avg_mae,
               "mae_std": std_mae,
               "mse_avg": avg_mse,
               "mse_std": std_mse}

    return results


df_werk = pd.read_csv("../../results/werk/results_werk_paper.csv")
no_text = compute_avg_std_results(df_werk, "-", 0)
bow50 = compute_avg_std_results(df_werk, "BoW", 50)
bow100 = compute_avg_std_results(df_werk, "BoW", 100)
bow500 = compute_avg_std_results(df_werk, "BoW", 500)
bong50 = compute_avg_std_results(df_werk, "BoNG", 50)
bong100 = compute_avg_std_results(df_werk, "BoNG", 100)
bong500 = compute_avg_std_results(df_werk, "BoNG", 500)
pv10 = compute_avg_std_results(df_werk, "PV", 10)
pv20 = compute_avg_std_results(df_werk, "PV", 20)
pv100 = compute_avg_std_results(df_werk, "PV", 100)
lda10 = compute_avg_std_results(df_werk, "LDA", 10)
lda20 = compute_avg_std_results(df_werk, "LDA", 20)
lda100 = compute_avg_std_results(df_werk, "LDA", 100)

baselines = [no_text, bow50, bow100, bow500, bong50, bong100, bong500, pv10, pv20, pv100, lda10, lda20, lda100]

tappbert_all = df_werk[df_werk["text_encoding"].str.contains("BERT")]
bert_encodings = list(dict.fromkeys([col for col in df_werk["text_encoding"].to_list() if col.startswith("BERT")]))  # keep order
# bert_encodings = np.unique([col for col in df_werk["text_encoding"].to_list() if col.startswith("BERT")])  # ignore order

avg_bert_accuracies = []
avg_bert_f1 = []
avg_bert_f1_macro = []
avg_bert_precisions = []
avg_bert_precisions_macro = []
avg_bert_recalls = []
avg_bert_recalls_macro = []
avg_bert_maes = []
avg_bert_mses = []
std_bert_accuracies = []
std_bert_f1 = []
std_bert_f1_macro = []
std_bert_precisions = []
std_bert_precisions_macro = []
std_bert_recalls = []
std_bert_recalls_macro = []
std_bert_maes = []
std_bert_mses = []

for bert_model in bert_encodings:
    bert_result = tappbert_all[tappbert_all["text_encoding"] == bert_model]

    bert_accuracies = bert_result["next_activity_acc"].to_numpy()
    bert_f1 = bert_result["next_activity_f1_weighted"].to_numpy()
    bert_f1_macro = bert_result["next_activity_f1_macro"].to_numpy()
    bert_precisions = bert_result["next_activity_precision_weighted"].to_numpy()
    bert_precisions_macro = bert_result["next_activity_precision_macro"].to_numpy()
    bert_recalls = bert_result["next_activity_recall_weighted"].to_numpy()
    bert_recalls_macro = bert_result["next_activity_recall_macro"].to_numpy()
    bert_maes = bert_result["next_time_mae"].to_numpy()
    bert_mses = bert_result["next_time_mse"].to_numpy()

    bert_accuracy_avg = np.mean(bert_accuracies)
    bert_f1_avg = np.mean(bert_f1)
    bert_f1_avg_macro = np.mean(bert_f1_macro)
    bert_precision_avg = np.mean(bert_precisions)
    bert_precision_avg_macro = np.mean(bert_precisions_macro)
    bert_recall_avg = np.mean(bert_recalls)
    bert_recall_avg_macro = np.mean(bert_recalls_macro)
    bert_mae_avg = np.mean(bert_maes)
    bert_mse_avg = np.mean(bert_mses)

    bert_accuracy_std = np.std(bert_accuracies)
    bert_f1_std = np.std(bert_f1)
    bert_f1_std_macro = np.std(bert_f1_macro)
    bert_precision_std = np.std(bert_precisions)
    bert_precision_std_macro = np.std(bert_precisions_macro)
    bert_recall_std = np.std(bert_recalls)
    bert_recall_std_macro = np.std(bert_recalls_macro)
    bert_mae_std = np.std(bert_maes)
    bert_mse_std = np.std(bert_mses)

    bert_accuracy_avg = round(bert_accuracy_avg, 4)
    bert_f1_avg = round(bert_f1_avg, 4)
    bert_f1_avg_macro = round(bert_f1_avg_macro, 4)
    bert_precision_avg = round(bert_precision_avg, 4)
    bert_precision_avg_macro = round(bert_precision_avg_macro, 4)
    bert_recall_avg = round(bert_recall_avg, 4)
    bert_recall_avg_macro = round(bert_recall_avg_macro, 4)
    bert_mae_avg = round(bert_mae_avg, 4)
    bert_mse_avg = round(bert_mse_avg, 4)

    bert_accuracy_std = round(bert_accuracy_std, 4)
    bert_f1_std = round(bert_f1_std, 4)
    bert_f1_std_macro = round(bert_f1_std_macro, 4)
    bert_precision_std = round(bert_precision_std, 4)
    bert_precision_std_macro = round(bert_precision_std_macro, 4)
    bert_recall_std = round(bert_recall_std, 4)
    bert_recall_std_macro = round(bert_recall_std_macro, 4)
    bert_mae_std = round(bert_mae_std, 4)
    bert_mse_std = round(bert_mse_std, 4)

    avg_bert_accuracies.append(bert_accuracy_avg)
    avg_bert_f1.append(bert_f1_avg)
    avg_bert_f1_macro.append(bert_f1_avg_macro)
    avg_bert_precisions.append(bert_precision_avg)
    avg_bert_precisions_macro.append(bert_precision_avg_macro)
    avg_bert_recalls.append(bert_recall_avg)
    avg_bert_recalls_macro.append(bert_recall_avg_macro)
    avg_bert_maes.append(bert_mae_avg)
    avg_bert_mses.append(bert_mse_avg)

    std_bert_accuracies.append(bert_accuracy_std)
    std_bert_f1.append(bert_f1_std)
    std_bert_f1_macro.append(bert_f1_std_macro)
    std_bert_precisions.append(bert_precision_std)
    std_bert_precisions_macro.append(bert_precision_std_macro)
    std_bert_recalls.append(bert_recall_std)
    std_bert_recalls_macro.append(bert_recall_std_macro)
    std_bert_maes.append(bert_mae_std)
    std_bert_mses.append(bert_mse_std)

#####
avg_accuracy = [r["accuracy_avg"] for r in baselines] + avg_bert_accuracies
avg_f1 = [r["f1_avg"] for r in baselines] + avg_bert_f1
avg_f1_macro = [r["f1_avg_macro"] for r in baselines] + avg_bert_f1_macro
avg_precision = [r["precision_avg"] for r in baselines] + avg_bert_precisions
avg_precision_macro = [r["precision_avg_macro"] for r in baselines] + avg_bert_precisions_macro
avg_recall = [r["recall_avg"] for r in baselines] + avg_bert_recalls
avg_recall_macro = [r["recall_avg_macro"] for r in baselines] + avg_bert_recalls_macro
avg_mae = [r["mae_avg"] for r in baselines] + avg_bert_maes
avg_mse = [r["mse_avg"] for r in baselines] + avg_bert_mses

std_accuracy = [r["accuracy_std"] for r in baselines] + std_bert_accuracies
std_f1 = [r["f1_std"] for r in baselines] + std_bert_f1
std_f1_macro = [r["f1_std_macro"] for r in baselines] + std_bert_f1_macro
std_precision = [r["precision_std"] for r in baselines] + std_bert_precisions
std_precision_macro = [r["precision_std_macro"] for r in baselines] + std_bert_precisions_macro
std_recall = [r["recall_std"] for r in baselines] + std_bert_recalls
std_recall_macro = [r["recall_std_macro"] for r in baselines] + std_bert_recalls_macro
std_mae = [r["mae_std"] for r in baselines] + std_bert_maes
std_mse = [r["mse_std"] for r in baselines] + std_bert_mses

df = pd.DataFrame({
    "model": ["no_text", "bow50", "bow100", "bow500", "bong50", "bong100", "bong500", "pv10", "pv20", "pv100",
              "lda10", "lda20", "lda100"] + bert_encodings,
    "accuracy_avg": avg_accuracy,
    "accuracy_std": std_accuracy,
    "f1_avg": avg_f1,
    "f1_std": std_f1,
    "f1_avg_macro": avg_f1_macro,
    "f1_std_macro": std_f1_macro,
    "precision_avg": avg_precision,
    "precision_std": std_precision,
    "precision_avg_macro": avg_precision_macro,
    "precision_std_macro": std_precision_macro,
    "recall_avg": avg_recall,
    "recall_std": std_recall,
    "recall_avg_macro": avg_recall_macro,
    "recall_std_macro": std_recall_macro,
    "mae_avg": avg_mae,
    "mae_std": std_mae,
    "mse_avg": avg_mse,
    "mse_std": std_mse,
})

df.to_csv("../../results/werk/results_werk_paper_avg_std.csv")
