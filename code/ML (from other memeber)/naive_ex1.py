# Import list
import numpy as np
from sklearn.datasets import fetch_20newsgroups # Data set
from sklearn.feature_extraction.text import CountVectorizer # for String data to bow (DTW)
from sklearn.feature_extraction.text import TfidfTransformer# for bow to (TF-IDF)
from sklearn.naive_bayes import MultinomialNB # navie_bayes model
from sklearn.metrics import accuracy_score # test accuracy_score

# Load data from sklearn (Twenty Newsgroups)
news_data = fetch_20newsgroups(subset='train')

# The new's Data is String data so it need to translate to DTM or TF_IDF

DTM_translate = CountVectorizer()
TF_IDF_translate = TfidfTransformer()

# Translate String Data. 
X_train_DTM = DTM_translate.fit_transform(news_data.data)
X_train_TF_IDF = TF_IDF_translate.fit_transform(X_train_DTM)

# Make naive_bayes model, and train X_train_TF_IDF (Can also use X_train_DTM)
model = MultinomialNB(alpha=1.0,class_prior=None,fit_prior=True)
model.fit(X_train_TF_IDF, news_data.target)

# Load test_dataset from sklearn
news_data_test = fetch_20newsgroups(subset='test', shuffle=True)

# Translate test_String Data. 
X_test_DTM = DTM_translate.transform(news_data_test.data)
X_test_TF_IDF = TF_IDF_translate.transform(X_test_DTM)

# Make test String from real_world car news. source : https://www.autocar.co.uk/car-review/ford/ranger-raptor/first-drives/ford-ranger-raptor-special-edition-2022-uk-review
test_string = np.array(["Do you ever find yourself looking at the hulking, jacked-up Ford Ranger Raptor pick-up truck and thinking \“it’s nice, but it’s not quite lairy enough\”? Us neither: its outlandish off- road suspension-and-tyre package already give it more than enough presence on Britain’s cramped streets (not to mention agreeable levels of countryside competence). But this new Special Edition which arrives as the current-generation Ranger prepares to bow out-ups the ante with “extra badass as standard”.Roughly translated, that means it gains racing stripes, red accents inside and out and matt-black trim all round. If it didn’t stick out in the supermarket car park before, you can guarantee that it will now."])

# Translate test String Data.
test_string_DTM = DTM_translate.transform(test_string)
test_string_TF_IDF = TF_IDF_translate.transform(test_string_DTM)

# Predicted test_string
test_predicted = model.predict(test_string_TF_IDF)

# Print test_string and predicted result.
print(test_string)
print("\n{} : {}\n".format(test_predicted[0],news_data.target_names[test_predicted[0]]))

# Predicted test dataset
Predicted = model.predict(X_test_TF_IDF)

# Test predicted dataset accuracy with orginal data
print("naive_bayes model accuracy: ", accuracy_score(news_data_test.target,Predicted))
