# Gene expression analysis for Asthma


## Requirements

* Python 3.6

Install all dependencies by calling 

    pip install -r requirements.txt

## Dataset

The dataset is collected in U-BIOPRED (Unbiased BIOmarkers in PREDiction of respiratory disease outcomes) study and can be downloaded from the following link
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE69683

There were 498 samples in total, split into four cohorts:
1. Healthy, non-smoking (87 samples)
2. Moderate asthma, non-smoking (77 samples)
3. Severe asthma, non-smoking (246 samples)
4. Severe asthma, smoking (88 samples)

This study focus on binary classification of non-smoking and smoking severe asthmatics. 
The purpose of experiment one was to determine if smoking caused significant enough effects on gene expression to allow high performance classification.


## Data Preprocessing

Convert the dataset to csv format 
   
    python process.py
    
## Training

    python main.py