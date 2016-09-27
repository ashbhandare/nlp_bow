import pandas as pd   


train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
print(train.shape)
print train["review"][6]
print "\n"+str(train["sentiment"][6])