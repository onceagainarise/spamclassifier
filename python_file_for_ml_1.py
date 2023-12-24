# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle


# %%
spam_mail_1=pd.read_csv("spam_ham_dataset.csv")
spam_mail_1.head()

# %%
columns_to_delete = ['Unnamed: 0', 'label_num']
spam_mail_1 = spam_mail_1.drop(columns=columns_to_delete)
string_to_delete = 'Subject:'
spam_mail_1['text'] = spam_mail_1['text'].str.replace(string_to_delete, '')

# %%
spam_mail_2 = pd.read_csv("spam_and_ham_file_2.csv",encoding='ISO-8859-1')
columns_delete = ['Unnamed: 2'	,'Unnamed: 3'	,'Unnamed: 4']
spam_mail_2= spam_mail_2.drop(columns=columns_delete)
spam_mail_2.rename(columns={'v1':'label','v2':'text'},inplace=True)
spam_mail_2.head()

# %%
mail_spam = pd.concat([spam_mail_1, spam_mail_2], ignore_index=True)

# Save the merged dataframe to a new CSV file
mail_spam.to_csv('merged_file.csv', index=False)
mail_spam.head(10)

# %%
print(mail_spam.shape)


# %%
mail_spam.loc[mail_spam['label'] == 'spam', 'label'] = 0
mail_spam.loc[mail_spam['label'] == 'ham', 'label'] = 1


# %%
X = mail_spam['text']
Y = mail_spam['label']


# %%
print(X)
print(Y)


# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print(X.shape)
print(X_train.shape)
print(X_test.shape)


# %%
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


# %%
print(X_train_features)


# %%
label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(Y_train)
Y_test = label_encoder.transform(Y_test)


# %%
model = LogisticRegression()
model.fit(X_train_features, Y_train)


# %%
prediction_on_training_data = model.predict(X_train_features)
accuracy_score_of_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data = ', accuracy_score_of_data)


# %%
# Use the already trained model for predictions on the test set
prediction_on_test_data = model.predict(X_test_features)
accuracy_score_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data = ', accuracy_score_on_test_data)


# %%

# %%
# Assuming you have already defined the 'feature_extraction' and 'model' objects
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
# Example input mail
"""input_mail = "ehronline web address change this message is intended for ehronline users only  "
input_data_feature = feature_extraction.transform([input_mail])

# Make a prediction using the trained model
prediction = model.predict(input_data_feature)

# Display the prediction
print("Predicted class:", prediction)

if prediction[0] == 0:
    print("This mail is a spam mail.")
else:
    print("This is not a spam mail.") """



