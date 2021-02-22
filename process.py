import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn.cluster as cluster
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Dataset link https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE69683
# import GEOparse
# gse = GEOparse.get_GEO(filepath="./GSE69683_family.soft.gz")


# label_list = ["cohort: Healthy, non-smoking", "cohort: Severe asthma, non-smoking"]
# labels = []

# print("GSM example:")
# for i, (gsm_name, gsm) in enumerate(gse.gsms.items()):
#     print("Name: ", gsm_name)
#     print("Metadata:",)
#     for key, value in gsm.metadata.items():
#         if key == 'characteristics_ch1' and value[0] in label_list:

#             labels.append(value[0])

#             # print ("Table data:",)
#             df = gsm.table

#             columns = df["ID_REF"].to_numpy()
#             # df = pd.DataFrame([['A', 1],['B',3]])
#             df = df.T
#             df.columns = df.iloc[0]
#             df = df.drop(df.index[0])

#             if i == 0:
#                 print(len(columns))
#                 h_sa_comb = pd.DataFrame(columns=columns)

#             h_sa_comb = h_sa_comb.append(df)

#             break

# h_sa_comb.index = labels

# h_sa_comb.to_csv("GSE69683_series.csv")

# print(h_sa_comb.shape)

# print(h_sa_comb)




h_sa_comb = pd.read_csv('GSE69683_series.csv', index_col='Unnamed: 0')


print(h_sa_comb)
# encode label to either 0 or 1
labelencoder_X = LabelEncoder()
h_sa_labels = labelencoder_X.fit_transform(np.array(h_sa_comb.index, dtype='str'))


# fill NaN values with mesn of each gene
h_sa_comb.fillna(h_sa_comb.mean())

nRow, nCol = h_sa_comb.shape
print(f'There are {nRow} samples and {nCol} features in dataframe.')


# split data into testing and training samples
train_x, test_x, train_y, test_y = train_test_split(h_sa_comb, h_sa_labels, test_size=0.2, random_state=0)


# scale data
sc = StandardScaler()
transformed_train_x = sc.fit_transform(train_x)
transformed_test_x = sc.fit_transform(test_x)

print(len(transformed_train_x), len(train_y))



# visulization
#Visualize data using Principal Component Analysis.
print("Principal Component Analysis (PCA)")
pca = PCA(n_components = 2).fit_transform(transformed_train_x)
pca_df = pd.DataFrame(data=pca, columns=['PC1','PC2'])
print(pca_df)


pca_df = pca_df.join(pd.DataFrame(data=train_y, columns=['Class']))

print(pca_df)

palette = sns.color_palette("muted", n_colors=2)
sns.set_style("white")
sns.scatterplot(x='PC1',y='PC2',hue='Class',data=pca_df, palette=palette, linewidth=0.2, s=30, alpha=1).set_title('PCA')
plt.show()


#Visualize data using t-SNE.
print("t-Distributed Stochastic Neighbor Embedding (tSNE)")
model = TSNE(learning_rate = 10, n_components = 2, random_state=123, perplexity = 30)
tsne = model.fit_transform(transformed_train_x)
tsne_df = pd.DataFrame(data=tsne, columns=['t-SNE1','t-SNE2']).join(pd.DataFrame(data=train_y, columns=['Class']))
palette = sns.color_palette("muted", n_colors=2)
sns.set_style("white")
sns.scatterplot(x='t-SNE1',y='t-SNE2',hue='Class',data=tsne_df, palette=palette, linewidth=0.2, s=30, alpha=1).set_title('t-SNE')
plt.show()




# feature extraction
test = SelectKBest(score_func=f_classif, k=12)
fit = test.fit(transformed_train_x, train_y)
# summarize scores
np.set_printoptions(precision=3)
# print(fit.scores_)
# print(len(fit.scores_))
selected_train_x = fit.transform(transformed_train_x)
selected_test_x = fit.transform(transformed_test_x)


# training
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

# prediction
pred_y = clf.predict(selected_test_x)

print(accuracy_score(test_y, pred_y))

cm = confusion_matrix(test_y, pred_y)

sns.heatmap(cm, center=True)
plt.show()