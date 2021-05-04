import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# load dataset
print("Loading dataset ...")
h_sa_comb = pd.read_csv('./dataset/GSE69683_series.csv', index_col='Unnamed: 0')

# encode label to either 0 or 1
labelencoder_X = LabelEncoder()
h_sa_labels = labelencoder_X.fit_transform(np.array(h_sa_comb.index, dtype='str'))

# split data into testing and training samples
train_x, test_x, train_y, test_y = train_test_split(h_sa_comb, h_sa_labels, test_size=0.2, random_state=0)

# scale data
sc = StandardScaler()
transformed_train_x = sc.fit_transform(train_x)
transformed_test_x = sc.transform(test_x)

# Visualize data using Principal Component Analysis.
print("Principal Component Analysis (PCA)")
pca = PCA(n_components=2).fit_transform(transformed_train_x)
pca_df = pd.DataFrame(data=pca, columns=['PC1', 'PC2'])
pca_df = pca_df.join(pd.DataFrame(data=train_y, columns=['Class']))

palette = sns.color_palette("muted", n_colors=2)
sns.set_style("white")
sns.scatterplot(x='PC1', y='PC2', hue='Class', data=pca_df, palette=palette, linewidth=0.2, s=30, alpha=1).set_title(
    'PCA')
plt.show()

# Visualize data using t-SNE.
print("t-Distributed Stochastic Neighbor Embedding (tSNE)")
model = TSNE(learning_rate=10, n_components=2, random_state=123, perplexity=30)
tsne = model.fit_transform(transformed_train_x)
tsne_df = pd.DataFrame(data=tsne, columns=['t-SNE1', 't-SNE2']).join(pd.DataFrame(data=train_y, columns=['Class']))
palette = sns.color_palette("muted", n_colors=2)
sns.set_style("white")
sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='Class', data=tsne_df, palette=palette, linewidth=0.2, s=30,
                alpha=1).set_title('t-SNE')
plt.show()

# feature extraction
print("Feature selection")
test = SelectKBest(score_func=f_classif, k=12)
fit = test.fit(transformed_train_x, train_y)
# summarize scores
np.set_printoptions(precision=3)
selected_train_x = fit.transform(transformed_train_x)
selected_test_x = fit.transform(transformed_test_x)

# training
print("Training with MLP")
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
print("Evaluating on test data")
pred_y = clf.predict(selected_test_x)
print(f"accuracy: {accuracy_score(test_y, pred_y)}")

# plot confusion matrix
cm = confusion_matrix(test_y, pred_y)
sns.heatmap(cm, center=True)
plt.show()
