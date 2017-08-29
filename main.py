## Challenge: Multi-class sentence classification
## author: Amir Abu Jandal

# Importing essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download()

# Reading training and test files to list data structures
xtrain = [line.rstrip('\n') for line in open('xtrain.txt', encoding="utf8")]
ytrain = [line.rstrip('\n') for line in open('ytrain.txt', encoding="utf8")]
xtest = [line.rstrip('\n') for line in open('xtest.txt', encoding="utf8")]

##[A] Data Preprocessing
# Since whitespaces were lost, so first task is word segmentation
# I will use Python's word segmentation library for this task
from wordsegment import segment
xtrain = [segment(line) for line in xtrain]
xtest = [segment(line) for line in xtest]

# Now saving these results to a text file,
# as the above instructions consumed allot of time
write_file = open('xtrain_segments.txt', 'w')
for line in xtrain:
    write_file.write("%s\n" % line)
write_file = open('xtest_segments.txt', 'w')
for line in xtest:
    write_file.write("%s\n" % (' ').join(line)) 
    
# Reading the above saved files back in Python
read_file = open('xtrain_segments.txt', encoding="utf8")
xtrain_segmented = [line.rstrip('\n') for line in read_file]
read_file = open('xtest_segments.txt', encoding="utf8")
xtest_segmented = [line.rstrip('\n') for line in read_file]

# Now let us tokenize each sentence
from nltk.tokenize import word_tokenize
xtrain_segmented = [word_tokenize(line) for line in xtrain_segmented]
xtest_segmented = [word_tokenize(line) for line in xtest_segmented]

# Now let us remove the stop words from each line
from nltk.corpus import stopwords
english_stops = stopwords.words('english')
xtrain_filtered = []
xtest_filtered = []
for line in xtrain_segmented:
    xtrain_filtered.append([word for word in line if not word in english_stops])
for line in xtest_segmented:
    xtest_filtered.append([word for word in line if not word in english_stops])

# Now the last challenge in preprocessing is compiling the training data set
# As the xtrain and ytrain are not of the same size, therefore ytrain should be 
# subsetted equal the size of xtrain. Note: There is 1 to 1 correspondence in both.
ytrain_subsetted = ytrain[0: len(xtrain_filtered)]

# Finally, it is time to make a training_dataset and write results to a file
training_dataset = []
i = 0
for sentence in xtrain_filtered:
    training_dataset.append((list(sentence), ytrain_subsetted[i]))    
    i = i + 1    

# Converting it to a pandas dataframe
dataset = pd.DataFrame(data=training_dataset)
dataset.columns = ["Sentences", "Novels"]
dataset["Sentences"] = [(" ").join(row) for row in dataset["Sentences"]]

##[B] Text to a Feature representation
# Let us consider CountVectorizer from sklearn
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=10000, lowercase=False)
X_train_tr = vectorizer.fit_transform(dataset["Sentences"])
print(X_train_tr.shape)

# Now, the dataset should be split in to train and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
       X_train_tr, dataset["Novels"], test_size=0.2, random_state=0)

# Thanks to sklearn, let us quickly train some multinomial models
# Model Training: Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score
mnb_classifier = MultinomialNB()
mnb_classifier.fit(X_train, y_train)
model_accuracies = cross_val_score(estimator=mnb_classifier, 
                                   X=X_train, y=y_train, cv=5) 
model_accuracies.mean()
model_accuracies.std()
# Model Testing: Multinomial Naive Bayes
from sklearn import metrics
y_pred = mnb_classifier.predict(X_test)
metrics.confusion_matrix(y_test, y_pred)
metrics.f1_score(y_test,y_pred)
test_accuracy = metrics.accuracy_score(y_test, y_pred)
print("Multinomial Naive Bayes Classifier Test Accuracy: ", test_accuracy*100)

# Making predictions and writing results to output file
xtest_filtered = [(" ").join(row) for row in xtest_filtered]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=10000, lowercase=False)
X_test_tr = vectorizer.fit_transform(xtest_filtered)
y_pred = mnb_classifier.predict(X_test_tr)
# Writing results to output file: ypred.txt
write_file = open('y_pred.txt', 'w')
for line in y_pred:
    write_file.write("%s\n" % line)

# Model Training: Random Forests Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
rf_classifier = RandomForestClassifier(n_estimators=200, 
                                        criterion='entropy', random_state=0)
rf_classifier.fit(X_train, y_train)
model_accuracies = cross_val_score(estimator=rf_classifier, 
                                   X=X_train, y=y_train, cv=5) 
model_accuracies.mean()
model_accuracies.std()

# Model Testing: Random Forests Classifier
from sklearn import metrics
y_pred = rf_classifier.predict(X_test)
metrics.confusion_matrix(y_test, y_pred)
test_accuracy = metrics.accuracy_score(y_test, y_pred)
print("Random Forests Test Accuracy: ", test_accuracy*100)

# Model Training: SVMs
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
svc_classifier = SVC(kernel='linear', random_state=0)
svc_classifier.fit(X_train, y_train)
model_accuracies = cross_val_score(estimator=svc_classifier, 
                                   X=X_train, y=y_train, cv=5) 
model_accuracies.mean()*100
model_accuracies.std()*100

# Model Testing: SVMs
from sklearn import metrics
y_pred = svc_classifier.predict(X_test)
metrics.confusion_matrix(y_test, y_pred)
test_accuracy = metrics.accuracy_score(y_test, y_pred)
print("SVMs Test Accuracy: ", test_accuracy*100)

# Model Training: ANNs
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Setting target variable as dummy variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
enc = OneHotEncoder(sparse=True) # Key here is sparse=False!
dummy_y = enc.fit_transform(y_train.reshape((y_train.shape[0]),1))

# Initialize the ANN
ann_classifier = Sequential()

# Adding the input layer and the first hidden layer
no_input_features = X_train_tr.shape[1]
ann_classifier.add(Dense(units=64, kernel_initializer='uniform', 
                         activation='relu', input_dim=no_input_features))
ann_classifier.add(Dropout(0.4))

# Adding another hidden layer
ann_classifier.add(Dense(units=32, kernel_initializer='uniform', 
                         activation='relu'))
ann_classifier.add(Dropout(0.1))

# Adding the output layer
ann_classifier.add(Dense(units=12, kernel_initializer='uniform', 
                         activation='softmax'))

# Compiling the ANN
ann_classifier.compile(optimizer='adam', loss='categorical_crossentropy', 
                       metrics=['accuracy'])

# Fitting the ANN to the training set
def samples(x_source, y_source, size):
    while True:
        for i in range(0, x_source.shape[0], size):
            j = i + size
            
            if j > x_source.shape[0]:
                j = x_source.shape[0]
                
            yield x_source[i:j].toarray(), y_source[i:j].toarray()
            
ann_classifier.fit_generator(samples(X_train, dummy_y, 10),X_train.shape[0], nb_epoch=2, verbose=1)

# Evaluating ann_classifier
dummy_y_test = enc.fit_transform(y_test.reshape((y_test.shape[0]),1))
scores = ann_classifier.evaluate(X_test, dummy_y_test)
print("\n%s: %.5f%%" % (ann_classifier.metrics_names[1], scores[1]*100))

from sklearn.cross_validation import KFold
kfold = KFold(n_folds=5, shuffle=True, random_state=0)
results = cross_val_score(ann_classifier, X_train, dummy_y, cv=kfold)
print("Baseline: %.5f%% (%.5f%%)" % (results.mean()*100, results.std()*100))

# Predicting given test data (xtest)
xtest_filtered = [(" ").join(row) for row in xtest_filtered]
vectorizer = CountVectorizer(max_features=10000, lowercase=False)
X_test_tr = vectorizer.fit_transform(xtest_filtered)
y_pred = le.inverse_transform(ann_classifier.predict(X_test_tr))

# Writing results to output file: ypred.txt
write_file = open('y_pred.txt', 'w')
for line in y_pred:
    write_file.write("%s\n" % line)
    
    