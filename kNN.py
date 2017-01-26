# Importing standard libraries
import sys
import math
import random
import collections

# Class for data contents
class dataContents(object):
	# Defining instance members attributes and instances
	def __init__(self,name):
		self.attributes = None
		self.getAttributes(name)
		self.instances = None
		self.getInstances(name)
		
		
	# Method to get attributes from file 
	def getAttributes(self,name):
		attrDict = {}
		numAttr = 0
		f = open(name)
		for line in f:
			if line[0] != '@':
				break
			elif '@relation' in line or '@data' in line:
				continue
			else:
				newline = line[len('@attribute'):]
				lineList = newline.split(" ")
				
				if '{' in lineList[2] or '}' in lineList[2]:
					for i in range(len(lineList[2])):
						if lineList[2][i] == '{':
							startInd = i
						if lineList[2][i] == '}':
							endInd = i
					attrStr = lineList[2][0:startInd] + lineList[2][startInd+1: endInd] + lineList[2][endInd+1:]
					attributeVals = [elem.strip() for elem in attrStr.split(",")]
				else:
					attributeVals = lineList[2].strip()
					
				attrDict[numAttr] = {lineList[1]:attributeVals}
				numAttr += 1

		self.attributes = attrDict
		f.close()


		
	# Method to get instances from file
	def getInstances(self,name):
		f  = open(name)
		num = 0
		instanceList = []
		for line in f:
			if line[0] == '@':
				continue
			elif line == None:
				break
			else:
				num += 1
				allFeat = line.strip().split(',')
				numFeat = allFeat[0:-1]
				numFeatFl = [float(i) for i in numFeat] 
				inst = numFeatFl
				inst.append(allFeat[-1])
				instanceList.append(inst)
		self.instances = instanceList
		f.close()

		
# Method to get contents from file name
def getContents(name):
	contents = dataContents(name)
	return contents

	

# Class kNN
class kNN(object):

	# Defining class variables
	def __init__(self):
		self.train_set = None
		self.test_set = None
		self.k = None
		# Obtaining command line arguments
		self.obtainCLIArgs()
		print 'k value : ' + str(self.k)
	
		
		# Obtaining last attribute type
		numTrainAtt = len(self.train_set.attributes)
		for k,v in self.train_set.attributes[numTrainAtt-1].iteritems():
			lastAttName = k
			attrVals = v
		
		
		numCorrectCl = 0
		MAE = 0
		for testInsInd in range(len(self.test_set.instances)):
			testIns = self.test_set.instances[testInsInd]
			testInsX = testIns[0:-1]
			testInsY = testIns[-1]
			
			# Initializing k best
			kBestDists = []
			kBestDict = {}
			
				
			
			# Updating k best 
			for i in range(len(self.train_set.instances)):
				dist = self.computeDist(testInsX,self.train_set.instances[i][0:-1])
				
				
				if len(kBestDists) == 0:
					maxbest = 100000
				else:
					maxbest = max(kBestDists)


					
				if len(kBestDists) < self.k:
					
					if dist < maxbest:
					
						# add this dist to the dict
						try:
							d = kBestDict[dist]
							kBestDict[dist][i] = self.train_set.instances[i][-1]
						except KeyError:
							kBestDict[dist] = {}
							kBestDict[dist][i] = self.train_set.instances[i][-1]
						
						# add to the list
						try:
							ind = kBestDists.index(dist)
						except ValueError:
							kBestDists.append(dist)
							
					elif dist == maxbest:
					
						# add this dist to the dict
						try:
							d = kBestDict[dist]
							kBestDict[dist][i] = self.train_set.instances[i][-1]
						except KeyError:
							kBestDict[dist] = {}
							kBestDict[dist][i] = self.train_set.instances[i][-1]
				
					else:
						# add this dist to the dict
						try:
							d = kBestDict[dist]
							kBestDict[dist][i] = self.train_set.instances[i][-1]
						except KeyError:
							kBestDict[dist] = {}
							kBestDict[dist][i] = self.train_set.instances[i][-1]
						
						# add this to the list
						kBestDists.append(dist)
						
						
				elif len(kBestDists) == self.k:
					
					if dist < maxbest:
						# Replace in the dict
						try:
							d = kBestDict[dist]
							kBestDict[dist][i] = self.train_set.instances[i][-1]
						except KeyError:

							del kBestDict[maxbest]
							kBestDict[dist] = {}
							kBestDict[dist][i] = self.train_set.instances[i][-1]
							# Replace in the list
							kBestDists.remove(maxbest)
							kBestDists.append(dist)
												
					elif dist == maxbest:
						# add this dist to the dict
						try:
							d = kBestDict[dist]
							kBestDict[dist][i] = self.train_set.instances[i][-1]
						except KeyError:
							kBestDict[dist] = {}
							kBestDict[dist][i] = self.train_set.instances[i][-1]
						
						
			# Obtaining prediction
			if lastAttName == 'response':
				predY = self.regress(kBestDists, kBestDict)
			elif lastAttName == 'class':
				attrValIndMap = {}
				for i in range(len(attrVals)):
					attrValIndMap[attrVals[i]] = i
				predY = self.classify(kBestDists, kBestDict, attrVals, attrValIndMap, testInsInd)
				if predY == testInsY:
					numCorrectCl += 1
					
						
				
			# Print result
			if lastAttName == 'response':
				testInsY = float(testInsY)
				MAE += abs(testInsY-predY)
				print 'Predicted value : '+ '%.6f'%predY + '	Actual value : '+ '%.6f'%testInsY
			elif lastAttName == 'class': 
				print 'Predicted class : '+ str(predY)+'	Actual class : '+ str(testInsY)


		if lastAttName == 'response':
			MAE = MAE/len(self.test_set.instances)
			print 'Mean absolute error : ' + '%.16f'%MAE
			print 'Total number of instances : '+ str(len(self.test_set.instances))
		elif lastAttName == 'class':
			print 'Number of correctly classified instances : '+ str(numCorrectCl)
			print 'Total number of instances : '+ str(len(self.test_set.instances))
			accuracy = float(numCorrectCl)/len(self.test_set.instances)
			print 'Accuracy : '+'%.16f'%accuracy
		

	# Method for classification
	def classify(self,kBestDists,kBestDict, attrVals, attrValIndMap, testInsInd):
		# Count votes for labels
		yvalcount = []
		
		for i in range(len(attrVals)):
			yvalcount.append(0)

		kBestDists.sort()
		numNNConsidered = 0
		done = False
		labelList = []
		
		for i in range(len(kBestDists)):
			dist = kBestDists[i]	
			d = collections.OrderedDict(sorted(kBestDict[dist].items()))
			for k1,v1 in d.iteritems():
				labelList.append(v1)
				numNNConsidered += 1
				yvalcount[attrValIndMap[v1]] += 1
				if numNNConsidered == self.k:
					done = True
					break
			if done:
				break
		
		
		
		# Finding class label
		maxVote = max(yvalcount)
		found = False
		unique = True
		
		for i in range(len(yvalcount)):
			if found  and yvalcount[i] == maxVote:
				unique = False
				break
			if not found and yvalcount[i] == maxVote:
				found = True
				firstLblInd = i
				
		if unique:		
			predY = attrVals[yvalcount.index(maxVote)]
		else:	
			predY = attrVals[firstLblInd]

		return predY
		
	
	# Method for regression 
	def regress(self,kBestDists,kBestDict):

		kBestDists.sort()
		numNNConsidered = 0
		done = False
		sum = 0
		
		for i in range(len(kBestDists)):
			dist = kBestDists[i]	
			d = collections.OrderedDict(sorted(kBestDict[dist].items()))
			for k1,v1 in d.iteritems():
				sum += float(v1)
				numNNConsidered += 1
				if numNNConsidered == self.k:
					done = True
					break
			if done:
				break
		
		predY = sum/self.k
		return predY
		
		
	# Computing Euclidean distance
	def computeDist(self, feavec1,feavec2):
		dist = 0
		for i in range(len(feavec1)):
			dist +=(feavec1[i]-feavec2[i])**2
		dist = dist**(0.5)	
		return dist
		
		
	# Method to obtain data from command line arguments
	def obtainCLIArgs(self):
		
		trainFileName = sys.argv[1]
		testFileName = sys.argv[2]
		self.k = int(sys.argv[3])
		self.train_set = getContents(trainFileName)  
		self.test_set = getContents(testFileName)
		
		
kNN()	
