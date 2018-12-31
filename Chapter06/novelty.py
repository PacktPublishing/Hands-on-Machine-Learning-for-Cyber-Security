import sys
import os
sys.path.append('.')
sys.path.insert(0, os.getcwd())
sys.path.append('/usr/lib/python2.6/site-packages')

import math


#fhistory of privileges used

input_path_of_file_hist = "/datasets/historical.data"
data_raw_hist = sc.textFile(input_path_of_file_hist, 12)

#for each privilge a rarity map is present

rarity_map = {}
input_path_of_file_rare = "/datasets/rare.data"
data_raw_rare = sc.textFile(input_path_of_file_rare, 12)
arr = data_raw_rare.split(',')
privilege = arr[0] 
rarityscore = arr[1]
rarity_map[privilege] = rarityscore


priv_hist = {}
FOREACH line in data_raw_hist :
	if line in priv_hist:
		do_nothing = 1
	else:
		priv_hist[line] = 1


input_path_of_file_curr = "/datasets/current.data"
data_raw_curr = sc.textFile(input_path_of_file_curr, 12)

num_lines = sum(1 for line in open(input_path_of_file_curr))

FOREACH line in data_raw_curr :
	if line in priv_hist
		print "i dont care this is privilege is old"
	else:
		print "new activity detected"
		C = computeScore()
		score = C.compute(line,num_lines)




class NoxScoring():
    def __init__(self):
        self.item_raririty_table = []
	self.item_raririty_table.append([.8,1,0.1])
        self.item_raririty_table.append([.7,.8,0.2])
        self.item_raririty_table.append([.6,.7,0.3])
        self.item_raririty_table.append([.5,.6, 0.4])
        self.item_raririty_table.append([.4,.5, 0.5])
        self.item_raririty_table.append([.3, .4, 0.6])
        self.item_raririty_table.append([.2, .3, 0.7])
        self.item_raririty_table.append([.1, .2, 0.8])
        self.item_raririty_table.append([.001, .1, 0.9])
        self.item_raririty_table.append([0, .001, 1])	

    def threat_anomaly_score(self,rarityscore,totalusers):
	if rarityscore is None :
        	age = .9
	else :
		age = float(rarityscore) / float(totalusers)

        for row in self.item_raririty_table:
            if (age>=row[0]) and (age<row[1]):
                score = row[2]
                return score
        return score

    def combine_threat_score(self,score,correlationscore):
        combined_score = score * 1
        return combined_score

class computeScore:
    def __init__(self,userandkey,rarity):
        self.userandkey = userandkey
        self.anomaly_score = 0


	def compute(line,num_lines)
	total=num_lines
    itemrarity = rarity_map[line]
	T = NoxScoring()
	anomaly_score = T.threat_anomaly_score(int(itemrarity),int(total))
	return anomaly_score







