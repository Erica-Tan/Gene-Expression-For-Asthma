import pandas as pd
import GEOparse

gse = GEOparse.get_GEO(filepath="./dataset/GSE69683_family.soft.gz")

label_list = ["cohort: Healthy, non-smoking", "cohort: Severe asthma, non-smoking"]
labels = []

print("GSM example:")
for i, (gsm_name, gsm) in enumerate(gse.gsms.items()):
    print("Name: ", gsm_name)
    print(gsm.metadata)

    for key, value in gsm.metadata.items():
        if key == 'characteristics_ch1' and value[0] in label_list:

            labels.append(value[0])

            df = gsm.table

            columns = df["ID_REF"].to_numpy()
            # df = pd.DataFrame([['A', 1],['B',3]])
            df = df.T
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])

            if i == 0:
                print(len(columns))
                h_sa_comb = pd.DataFrame(columns=columns)

            h_sa_comb = h_sa_comb.append(df)


print("Finish ...")

h_sa_comb.index = labels

nRow, nCol = h_sa_comb.shape
print(f'There are {nRow} samples and {nCol} features in dataframe.')

# save csv dataset
print("Saving csv file")
h_sa_comb.to_csv("./dataset/GSE69683_series.csv")