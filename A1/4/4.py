from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

import json
import csv
#import pprint

#function to create dataset using json data
def createDataSet(x,ing,data,count):
	for row in data:
		a=[False]*count
		for i in row["ingredients"]:
			if i in ing:
				a[ing[i]]=True
		x.append(a)


#function to write output file
def writeCSV(ids,a,name):
	row=['id','cuisine']
	with open(name, 'w') as writeFile:
		writer = csv.writer(writeFile)
		writer.writerow(row)
		for i in range(0,len(ids)):
			row=[ids[i],a[i]]
			writer.writerow(row)
	writeFile.close()


#function for Decision Trees
def usingDT(train,test,count,labels,ids):
	clf=tree.DecisionTreeClassifier()
	clf=clf.fit(train,labels)
	a=clf.predict(test)
	writeCSV(ids,a,"DT_1.csv")


#function for KNN
def usingKNN(train,test,count,labels,ids):
	knn = KNeighborsClassifier(120)
	knn.fit(train,labels)
	a=knn.predict(test)
	writeCSV(ids,a,"KNN_1.csv")

	knn1 = KNeighborsClassifier(75)
	knn1.fit(train,labels)
	a=knn1.predict(test)
	writeCSV(ids,a,"KNN_2.csv")


#function for Naive Bayes
def usingNaiveBayes(train,test,count,labels,ids):

	mnb = MultinomialNB(0.1)
	mnb.fit(train,labels)
	a=mnb.predict(test)
	writeCSV(ids,a,"NB_1.csv")



if __name__ == "__main__":
	with open('train.json') as f:
	    data = json.load(f)
	labels=[]		#stores class labels
	label_set=set()	
	ids=[]			#stores ids
	ing=set()		#set of ingredients
	ingr={}			#dictionry of ingredient with corresponding index no
	count=0
	for row in data:
		labels.append(row["cuisine"])
		label_set.add(row["cuisine"])
		for i in row["ingredients"]:
			ing.add(i)

	for i in ing:
		ingr[i]=count
		count+=1
	f.close()

	training_set=[]
	createDataSet(training_set,ingr,data,count)	#create training set
	print("Size of Training Set: ",len(training_set))

	with open('test.json') as f:
	    data = json.load(f)
	for row in data:	    	
		ids.append(row["id"])
	f.close()
	test_set=[]
	createDataSet(test_set,ingr,data,count)		#create test set
	print("Size of Test Set: ",len(test_set))
	print("Number of Classes: ",len(label_set))

	#call decision tree
	usingDT(training_set,test_set,count,labels,ids)	
	#call naive bayes
	usingNaiveBayes(training_set,test_set,count,labels,ids)
	#call KNN
	#usingKNN(training_set,test_set,count,labels,ids)