# Import list
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

%matplotlib inline

#load dataset.
Data = pd.DataFrame(pd.read_csv("../[01]data_set/StudentsPerformance.csv"))

# first you need to download data set file. check [01]data_set/StudentsPerformance_csv_download.txt

#Set Columns
Data.columns = ['1','2','3','4','5','math','read','write']

#load dataset_1.
Data_1 = pd.DataFrame(pd.read_excel("../[01]data_set/Data.xlsx"))
#Set Columns
Data_1.columns = ['sales','result']

# Recommended for use in Jupyter Notebooks
#Set Graph data.

plt.figure(figsize=(10, 10))         
plt.scatter(Data.math, Data.write) 
plt.xlabel("math")    #label set              
plt.ylabel("read")                  
plt.grid()                          
plt.show()

plt.figure(figsize=(10, 10))         
plt.scatter(Data_1.sales, Data_1.result) 
plt.xlabel("sales")    #label set              
plt.ylabel("result")                  
plt.grid()                          
plt.show()                            

#calculate spearman result.
spearman_mw = stats.spearmanr(Data.math,Data.write)
spearman_sr = stats.spearmanr(Data_1.sales,Data_1.result)


print("spearman result \n math-write ")
print(spearman_mw)
print("spearman result \n sales-result")
print(spearman_sr)
