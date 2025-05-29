import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf

def plot_label_distribution(df):
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    label_counts = df[label_cols].sum().sort_values(ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x=label_counts.index, 
                y=label_counts.values, 
                palette="coolwarm", 
                hue=label_counts.index, 
                legend=False)
    plt.title("Label Distribution")
    plt.ylabel("Number of Comments")
    plt.xlabel("Labels")
    plt.show()


def plot_labels_per_comment(df):
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    label_sums = df[label_cols].sum(axis=1)

    # print(label_sums)
    plt.figure(figsize=(10,6))
    sns.countplot(x=label_sums)
    plt.title("Number of Labels per Comment")
    plt.xlabel("Number of Labels")
    plt.ylabel("Number of Comments")
    plt.show()


def plot_comment_length(df, text_column='comment_text'):
    df['length'] = df[text_column].astype(str).apply(len)

    plt.figure(figsize=(10,6))
    sns.histplot(df['length'], bins=50, kde=True, color='purple')
    plt.title("Distribution of Comment Length")
    plt.xlabel("Length of Comment(Charchaters)")
    plt.ylabel("Number of comments")
    plt.show()


def plot_label_correlation(df):
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    corr = df[label_cols].corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Between Labels")
    plt.show()



def plot_length_vs_toxicity(df):
    df_copy = df.copy()
    df_copy['length'] = df_copy['comment_text'].astype(str).apply(len)

    plt.figure(figsize=(10,6))
    sns.boxplot(x='toxic', y='length', data=df_copy)
    plt.title("Comment Length vs Toxicity")
    plt.xlabel("Toxic (0 = Clean, 1 = Toxic)")
    plt.ylabel("Comment Length (Characters)")
    plt.xticks([0, 1], ['Clean', 'Toxic'])
    plt.show()


def prepare_and_split(df):
    df = df.copy()
    # Drop comment_text
    df.drop(columns=['comment_text'], inplace=True)

    # Drop rows with empty
    df = df[df['clean_text'].notna() & (df['clean_text'].str.strip() != '')]

    df = shuffle(df, random_state=42).reset_index(drop=True)


    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    X = df['clean_text']
    Y = df[label_cols]

    # 80% train, 10%, 10%
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test




def plot_actual_vs_predicted(Y_true, Y_pred, label_names):
    true_counts = np.sum(Y_true, axis=0)
    pred_counts = np.sum(Y_pred, axis=0)

    df_plot = pd.DataFrame({
        'Label': label_names,
        'Actual': true_counts,
        'Predicted': pred_counts
    })

    df_plot.set_index('Label').plot(kind='bar', figsize=(10, 5))
    plt.title('Actual vs Predicted Label Counts')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()



class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, Y_val):
        self.X_val = X_val
        self.Y_val = Y_val
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        val_preds = (self.model.predict(self.X_val) > 0.5).astype(int)
        f1 = f1_score(self.Y_val, val_preds, average='macro')
        self.f1_scores.append(f1)
        print(f"\nEpoch {epoch+1}: val_f1_score: {f1:.4f}")

def plot_f1_history(f1_scores):
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(f1_scores)+1), f1_scores, marker='o')
    plt.title('Validation F1-Score per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.grid()
    plt.show()


# LSTM (processed|un-processed) # https://drive.google.com/file/d/1Y0gcbDBkBqEBlfamKrDkSPMFv798JjUu/view?usp=sharing 
# LSTM(With DS3) # https://drive.google.com/file/d/1yytbLL-VqD44YBqBTeuskTaezYPnOwpM/view?usp=sharing


# LSTM # https://colab.research.google.com/drive/1YMCrrDHdP8FeF1N-wjE8SKLbqDWlFIPK?usp=sharing
# BERT # https://colab.research.google.com/drive/1E0wgqTQPMU5Cja0e_k_jyGv-d15HiVlg?usp=sharing
# Roberta # 
