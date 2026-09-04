A baseline fraud classifier on the ULB credit card dataset

Accuracy is not a useful measure for this dataset since the fraud rate is so low to start, even if nothing is flagged as fraud by a model the accuracy is still high by the virtue of having large TN.

The output should show the average precision of 0.7 but the analysis of the test set should show 0.3 for precision and 0.9 for recall for the 0.5% transactions we set aside for "human eyes". Recall being high here is actually expected, since 0.5% of the transactions we pull up to look for the frauds that occur 0.17% of the times is actually a large sample.


The dataset can be downloaded from Kaggle to the data/ directory:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


To run:

pip install -r requirements.txt
python data/download.py
python run.py.

Status:
- primary investigation done
- next step: 
    -- try multiple rounds of training on the data and observe what sort of degeneracy develops
    -- simulate deliberately letting some fraudulent transactions through