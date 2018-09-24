# Penn-Paraphrase-Relationship
Determine the Relationship between two partial phrases

* Developed in **Python 2.7.15**
* Dependencies:
  * **numpy==1.15.1**
  * **pandas==0.23.4**
  * **scikit-learn==0.19.2**
  * **gensim==3.6.0**
  * **imbalanced-learn==0.3.3**
* Techniques used:
  * **Word2Vec**
  * **Doc2Vec**
  * **Principal Component Analysis**
  * **Synthetic Minority Over-sampling Technique (SMOTE)**
  * **Random-Forest Classifier**

* Steps:
  * Numeric values extracted across the columns.
  * Dropped the columns with only one distinct value.
  * **Word vectors** for each word within a phrase along with its **Doc vector** are extracted and terminated into a single         vector (for a single phrase) by taking average of **Words-Doc vectors**, followed by calculating similarity measurement         between **Source** and **Target** phrases.
  * **Synthetic Oversampling** applied on the training set, as the labels are not balanced.
  * **Dimensionality Reduction** technique is applied on continuous variables.
  * **Random Forest** classifier with **Grid Search** parameter tuning method is applied.
  
