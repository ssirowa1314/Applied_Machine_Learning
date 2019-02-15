# CS6510 HW 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import math as m
import copy as c

# Enter You Name Here
myname = "Doe-John-" # or "Doe-Jane-"

# Implement your decision tree below

class DecisionTree():
    def __init__(self):
        #self.root=Node()
        self.tree={}
        self.data={}
        self.nodeVal=0

    def learn(self, training_set, usedAttr_1={}, currNode=0,threshold=0.0):
        # implement this function
        mx=0.0
        mxAttr=-1           #index of best Attribute
        split=-1.0          #best split value
        l=len(training_set)
        usedAttr=c.deepcopy(usedAttr_1) 
        usedAttr_new=usedAttr
        h=self.computeGini(training_set);    #compute the entropy of currnode
        if(h<threshold):                             #make leaf if node is pure
            res=self.result(training_set)
            self.tree[currNode]=[-1,res]
            return
        for i in range(11):                     #loop to find best attribute
            if i not in usedAttr :              #check if a attribute is already used up in a branch
                a11=self.findSplit(training_set,i,h)        #find the split value for the attribute. returns a list 
                a1=a11[0]                                   #split value
                a3=a11[1]                                   #information gain
                if mx<a3 :                                  #better information gain, then choose the attribute
                    mx=a3;
                    mxAttr=i;
                    split=a1
        if(mxAttr>-1):                                      #if there is any information gain for a split
            usedAttr_new[mxAttr]=True                       #can't use split attribute again in the subbtrees
            l=self.nodeVal+1                                #left nodevalue 
            r=l+1;                                          #right nodevalue
            if len(usedAttr_new)<11:                        #if all attributes are used, then make leaf.
                self.tree[currNode]=[l,r]                   #assign left and right childs of a node 
                self.data[currNode]=[mxAttr,split]          #store data of a node
                self.nodeVal+=2                             #updated nodevalue
                left=[]                                     #will store left partition
                right=[]                                    #will store right partition
                self.partition(training_set, left, right, split, mxAttr)    #partition the current training set in left and right
                usedAttr_l=usedAttr_new                                     
                usedAttr_r=usedAttr_new
                self.learn(left,usedAttr_l,l,threshold)               #learn the left subtree
                self.learn(right,usedAttr_r,r,threshold)              #learn the right subree
            else:                                           #if all attributes used make leaf
                res=self.result(training_set)               #compute the label of leaf
                self.tree[currNode]=[-1,res]                #-1 as first child of node signifies it as leaf. second index store the label of the node
        else:                                               #if no information gain, then make leaf
            res=self.result(training_set)                   #compute the label of leaf
            self.tree[currNode]=[-1,res]                     #-1 as first child of node signifies it as leaf. second index store the label of the node



# implement this function. This function classifies a test data
    def classify(self, test_instance):
        result = 0 # baseline: always classifies as 0
        result=self.traverse(test_instance,0)               #traverse the tree
        return str(result)                                  #return the label


    def traverse(self,test_instance,node):                  #function to traverse the tree, to find the correct label
        if(self.tree[node][0] == -1):                       #if leaf return label.
            return self.tree[node][1]
        else:                                               #not leaf
            if float(test_instance[self.data[node][0]]) <= self.data[node][1]:      #go to left subtree
                return self.traverse(test_instance,self.tree[node][0])
            else:                                                                   #go to right subtree
                return self.traverse(test_instance,self.tree[node][1])


#this function computes the label of leaf
    def result(self,x):
        count_0=0
        count_1=1;
        for row in x:
            if row[11]=="1":
                count_1+=1
            else:
                count_0+=1

        return int(count_1>=count_0)        #return true if count_1 is bigger


#this function partitions a datase tinto left right. Stores the left partion in left and right in right
    def partition(self,x, left, right, split, i):   
        for row in x:
            if(float(row[i]) <= split):         #split based on split value
                left.append(row)
            else:
                right.append(row)


#function to find split value of a attribute. The gap b/w maximum and minimum attribute value is computed and 15 values are checked b/w max and min
    def findSplit(self,x,i,e):
        a=[]            #stores the attribute value
        l=len(x)
        a2=val=-1.0     #value of split
        mx=-1.0         #information gain
        for row in x:
            a.append(float(row[i]))
        a.sort()        #sort 
        gap=(a[l-1]-a[0])/15.0
        for k in range(0,15):
            count=0
            j=a[0]+gap*float(k+1)
            for it in a:
                if it<=j:
                    count+=1
                else:
                    break
            a1=self.computeGini(x,j,i)*(float(count)/l)+self.computeGini(x,j,i,1)*(float(l-count)/l)
            a2=e-a1
            if mx<a2:
                mx=a2
                val=j    
        return[val,mx]
        

#function to compute gini. p1 determines what is attribute index for which we are finding entropy. p2 determines the split value for the attribute. p3 determines wheter left subree or right subtree. x is training set
    def computeGini(self,x, p2=0.0, p1=-1, p3=0):
        count_1=0
        l=0;
        for row in x:
            if p1!=-1 and p3==0 and float(row[p1])<=p2:
                l+=1;
                if row[11] == "1":
                    count_1+=1
            elif p1!=-1 and p3==1 and float(row[p1])>p2:
                l+=1
                if row[11] == "1":
                    count_1+=1
            elif p1==-1:
                l+=1
                if row[11] == "1":
                    count_1+=1
        count_0=l-count_1
        #print(l,c)
        if(count_0==0 or count_1==0):
            #print("zero %d"%self.nodeVal)
            return 0.0
        pyes=float(count_1)/l
        pno=float(count_0)/l
        return (1-pyes*pyes-pno*pno)


    def retNodeval(self):
        return self.nodeVal


def run_decision_tree():

    # Load data set
    with open("wine-dataset.csv") as f:
        next(f, None)
        data = [tuple(line) for line in csv.reader(f, delimiter=",")]
    print ("Number of records: %d" % len(data))
    #print data

    # Split training/test sets
    # You need to modify the following code for cross validation.
    K = 10
    accuracy_total=0.0
    f = open(myname+"result.txt", "w")
    for j in range(10):
        training_set = [x for i, x in enumerate(data) if i % K != j]
        #print ("Size of training_set: %d" % len(training_set))
        test_set = [x for i, x in enumerate(data) if i % K == j]
        #print ("Size of test_set: %d" % len(test_set))
    
        # Construct a tree using training set
        tree = DecisionTree()
        tree.learn(training_set,{},0,0.17)
        print("Number of Nodes: ",tree.retNodeval())

        # Classify the test set using the tree we just constructed
        results = []
        for instance in test_set:
            result = tree.classify(instance)
            results.append( result == instance[-1])

        # Accuracy
        accuracy = float(results.count(True))/float(len(results))
        accuracy_total+=accuracy
        print("accuracy of fold %d: %.4f" %(j+1,accuracy))    

        # Writing results to a file (DO NOT CHANGE)
        f.write("accuracy of fold %d: %.4f\n" %(j+1,accuracy))
    accuracy=accuracy_total/10.0
    print("average accuracy: %.4f" %accuracy)
    f.write("average accuracy: %.4f\n" %accuracy)
    f.close()


if __name__ == "__main__":
    run_decision_tree()