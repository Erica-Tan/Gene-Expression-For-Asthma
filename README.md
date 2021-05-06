# Gene expression analysis for Asthma


## Requirements

* Python 3.6

Install all dependencies by calling 

    pip install -r requirements.txt

## Dataset

The dataset is collected in the U-BIOPRED (Unbiased BIOmarkers in PREDiction of respiratory disease outcomes) study and can be downloaded from the following link
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE69683

There were 498 samples in total, split into four cohorts:
1. Healthy, non-smoking (87 samples)
2. Moderate asthma, non-smoking (77 samples)
3. Severe asthma, non-smoking (246 samples)
4. Severe asthma, smoking (88 samples)

This study focuses on binary classification of non-asthmatics and severe non-smoking asthmatics. 
The purpose was to determine what differences in gene expression were present in severe asthmatics compared to health individuals

## Data Preprocessing

Convert the dataset to csv format 
   
    python convert_to_csv.py
    
## Training

    python main.py