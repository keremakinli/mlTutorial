from __future__ import division
import numpy as np
import pandas as pd
import csv
import collections
from sklearn.model_selection import train_test_split
#load the data
data = pd.read_csv("winequality-red.csv", header=None)
column_labels =data.loc[0,:]
column_labels_index={}
itr=0
for i in column_labels:
    column_labels_index[i]=itr
    itr+=1
rows=np.array(data.loc[1:])
rows=rows.astype(np.float)  #entire dataset
#rows=rows[:50]
rows_means={}               #dict for column means
for i in column_labels:
    rows_means[i]=0.0
means_list=np.mean(rows,axis=0) #mean of each col
itr=0
for i in rows_means:
    rows_means[i]=means_list[itr]
    itr+=1
rows_bin=[]
rows_bin_train=[]
rows_bin_test=[]
for i in rows:
    aa=np.greater_equal(i,means_list)
    rows_bin.append(aa.astype(int))
rows_bin=np.array(rows_bin)
print(collections.Counter(rows_bin[:, -1]))
rows_bin_train, rows_bin_test, y_train, y_test = train_test_split(rows_bin, rows_bin[:, -1], stratify=rows_bin[:, -1])
print(collections.Counter(rows_bin[:, -1]), collections.Counter(rows_bin_train[:, -1]))
'''
itr=0
for i in rows_bin:
    if itr<=len(rows_bin)*0.7:
        rows_bin_train.append(i)
    else:
        rows_bin_test.append(i)
    itr+=1
'''
print("rows_bin size: ",len(rows_bin))
#id3 starts here---
class Node:
    column=None
    isLeaf=False
    decision=None
    left,right=None,None
#print("rows_bin[:0]: ",rows_bin[:,0])
#print("type of rows_bin: ",type(rows_bin))

def makeTree(node,rows_bin_filtered,ancestral_cols ):
    rows_bin_filtered=np.array(rows_bin_filtered)
    #print("ancestral_cols: ",ancestral_cols)
    #print("len(rows_bin_filtered): ",len(rows_bin_filtered))
    if(len(ancestral_cols)==len(column_labels_index)-1):
        #print("rows_bin_filtered",rows_bin_filtered)
        node.isLeaf=True
        node.decision=0
        return node
    #base cases
    if rows_bin_filtered is None or len(rows_bin_filtered)==1 or len(rows_bin_filtered)==0:
        node.isLeaf=True
        if rows_bin_filtered is None or len(rows_bin_filtered)==0:
            node.decision=0
        else:
            node.decision=rows_bin_filtered[0,-1].astype(int)
        return node
    if sum(rows_bin_filtered[:,-1])==0:
        node.isLeaf=True
        node.decision=0
        return node
    if sum(rows_bin_filtered[:,-1]) == len(rows_bin_filtered[:,-1]) :
        node.isLeaf=True
        node.decision=1
        return node
    min_entropy_col,min_entropy=None,None
    for i in column_labels[:-1]:
        if i in ancestral_cols:
            continue    
        #make the set of different values for the ith col in rows_bin_filtered
        s_values=set(rows_bin_filtered[:,column_labels_index[i]])
        ss=0
        for ii in s_values:
            ii=ii.astype(int)
            #ii is the attribute instance
            y,n=0,0
            for jj in rows_bin_filtered:
                if jj[column_labels_index[i]]==ii:
                    if jj[-1]==0:
                        n+=1
                    elif jj[-1]==1:
                        y+=1
            p_y=y/(n+y)
            p_n=n/(n+y)
            ss+=( (p_y*np.log(p_y)/np.log(2))+(p_n*np.log(p_n)/np.log(2)) )*(n+y)
        ss=ss/len(rows_bin_filtered)
        if (min_entropy is None) or min_entropy > ss :
            min_entropy=ss
            min_entropy_col=i
    #print("min_entropy_col: ",min_entropy_col)
    node.column=min_entropy_col
    #recur for left and right children using node.column as differentiating attribute
    rows_bin_filtered_left=[]
    rows_bin_filtered_right=[]
    #print("lengths: ",len(rows_bin_filtered),", ",len(rows_bin_filtered_left),", ",len(rows_bin_filtered_right))
    
    for ii in rows_bin_filtered:
        if ii[column_labels_index[node.column]]==0:
            rows_bin_filtered_left.append(ii)
        else:
            rows_bin_filtered_right.append(ii)
    #print("lengths: ",len(rows_bin_filtered),", ",len(rows_bin_filtered_left),", ",len(rows_bin_filtered_right))
    
    #node.children=np.concatenate( (node.children ,[Node()] ))
    a=ancestral_cols
    a=np.concatenate((ancestral_cols,[node.column]))
    node.left=makeTree(Node(),rows_bin_filtered_left,a)
    #node.children=np.concatenate(( node.children ,[Node()] ))
    node.right=makeTree(Node(),rows_bin_filtered_right,a)
    #print("a: ",a)
    return node
root=Node()
print("Making tree")
root=makeTree(root,rows_bin_train,[])
print("Tree complete")
#print("root.left: ",root.left)
#print("root.right: ",root.right)
def traverse(i):
    node=root
    while node.isLeaf is False:
        if i[column_labels_index[node.column]]==0 and node.left is not None:
            node=node.left
        elif i[column_labels_index[node.column]]==1 and node.right is not None:
            node=node.right
        else:
            return 0    #random output for unseen examples
    return node.decision
def dfs(node,level=0):
    print("\t"*level," column: ",node.column)
    if node.isLeaf is True:
        print("decision: ",node.decision)
        return
    dfs(node.left,level+1)
    dfs(node.right,level+1)

print("root column label: ",root.column)
print("Printing Tree: ")
#dfs(root)
print("Tree printing complete ")

print("Testing: ")
e=0
f = dict()
f[0] = 0
f[1] = 0
print(collections.Counter(rows_bin_test[:, -1]))
for i in rows_bin_test:
    t=traverse(i)
    f[t] += 1
    #print("h: ",t," , y: ",i[-1])
    if t==i[-1]:
        e+=1
print(f)
print("e: ",e," accuracy: ",e/len(rows_bin_test))
