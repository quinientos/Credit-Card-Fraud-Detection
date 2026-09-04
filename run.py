import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_num_fraud_reviewable(y_true, scores, budget=0.005):
    """ Flag the top `budget` number of the rows, descending by score.
        Return precision, recall, and the number of frauds that would be reviewed.
    """
    y_true = np.array(y_true)
    k = int(len(scores) * budget)
    top_k = np.argsort(scores)[::-1][:k]
    caught = y_true[top_k].sum()
    
    return caught/k, caught/y_true.sum(), k


def main():
    # Data downloaded from Kaggle:
    # https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    df = pd.read_csv("data/creditcard.csv")

    # Take a look at the data first. 
    print(f"rows: {len(df)}  frauds: {df.Class.sum()}  rate: {df.Class.mean():.4%}")

    # Select the columns the model gets to see
    X = df.drop(columns=["Class", "Time"]).copy() # delete the clssifier info so we can properly test
    X["Amount"] = np.log1p(X["Amount"]) # log transform the amount column
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    # Check the data split
    # print(f"Training set: {y_train.sum()} frauds out of {len(y_train)} total")
    # print(f"Test set: {y_test.sum()} frauds out of {len(y_test)} total")
    # print(f"Test set fraud rate: {y_test.sum()/df['Class'].sum():.4%}")

    # transform the data and fit the model
    pipe = Pipeline([("scale", StandardScaler()),
                    ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))
                    ])
    pipe.fit(X_train, y_train)

    # Isolate the probabilities indicating fraud (vs. legitimate transactions)
    scores = pipe.predict_proba(X_test)[:,1]

    ap = average_precision_score(y_test, scores)
    precision, recall, k = get_num_fraud_reviewable(y_test, scores)

    print(f"average precision: {ap:.3f}")
    print(f"top {k} entries of {len(scores)}: precision {precision:.3f}, recall {recall:.3f}")


if __name__ == "__main__":
    main()