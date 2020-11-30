import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


raw_dataset = pd.read_csv('GSE69683_series.csv')
# raw_dataset.to_csv ('GSE69683_series.csv', index = None, header=True)



# strip out unneeded information and leave gene expression data, column labels
# and probe ids
raw_strimmed = raw_dataset.iloc[35:38, :].append(raw_dataset.iloc[61:, 0:])
raw_strimmed.columns = [raw_dataset.iloc[35, :]]
raw_strimmed.index = [raw_dataset.iloc[58:, 0]]
raw_strimmed = raw_strimmed.drop('!series_matrix_table_end')



# create dict for cohorts
columns = [col[0] for col in raw_strimmed.columns]
raw_strimmed.columns = columns
cohorts = list(set(columns))
cohorts.remove('!Sample_characteristics_ch1')
sorted_cohorts = {cohort: raw_strimmed[cohort] for cohort in cohorts}


# create data for Healthy and server asthma, transpose the dimention
h_ns = sorted_cohorts['cohort: Healthy, non-smoking'].iloc[3:,0:].apply( \
	pd.to_numeric, downcast='float').T

sa_ns = sorted_cohorts['cohort: Severe asthma, non-smoking'].iloc[3:,0:].apply( \
	pd.to_numeric, downcast='float').T

h_sa_comb = h_ns.append(sa_ns)


# encode label to either 0 or 1
labelencoder_X = LabelEncoder()
h_sa_labels = labelencoder_X.fit_transform(np.array(h_sa_comb.index, dtype='str'))


# fill NaN values with mesn of each gene
h_sa_comb.fillna(h_sa_comb.mean())

print(h_sa_comb.columns, h_sa_comb.index)


# split data into testing and training samples
train_x, test_x, train_y, test_y = train_test_split(h_sa_comb, h_sa_labels, test_size=0.2, random_state=0)


# scale data
sc = StandardScaler()
transformed_train_x = sc.fit_transform(train_x)
transformed_test_x = sc.fit_transform(test_x)

print(len(transformed_train_x), len(train_y))


# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# feature extraction
test = SelectKBest(score_func=f_classif, k=12)
fit = test.fit(transformed_train_x, train_y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
print(len(fit.scores_))
selected_train_x = fit.transform(transformed_train_x)
selected_test_x = fit.transform(transformed_test_x)






# prediction
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
clf = MLPClassifier(hidden_layer_sizes=10000,
					activation='identity',
					solver='lbfgs',
					alpha=0.1,
					batch_size=200,
					learning_rate='adaptive',
					max_iter=200,
					tol=0.00000001,
					verbose=True,
					early_stopping=False,
					validation_fraction=0.2).fit(selected_train_x, train_y)
pred_y = clf.predict(selected_test_x)

print(accuracy_score(test_y, pred_y))



import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(test_y, pred_y)

sns.heatmap(cm, center=True)
plt.show()