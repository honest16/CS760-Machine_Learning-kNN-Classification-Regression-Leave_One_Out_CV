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
	def __init__(self, train_set, test_set, k):
		self.train_set = train_set
		self.test_set = test_set
		self.k = k
		self.kmax = None
		self.numInCl = None
		
		
		# Identify whether k passed in is a scalar or a vector
		if type(self.k) == int:
			self.kmax = self.k
		elif len(self.k) > 1:
			self.kmax = max(self.k)
		
		
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
				

				
				if len(kBestDists) < self.kmax:
					
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
						
						
				elif len(kBestDists) == self.kmax:
					
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
				if type(self.k) is not int:
					self.numInCl = []
					for p in range(len(self.k)):
						self.numInCl.append(abs(float(testInsY)-predY[p]))
					
			elif lastAttName == 'class':
				attrValIndMap = {}
				for i in range(len(attrVals)):
					attrValIndMap[attrVals[i]] = i
				predY = self.classify(kBestDists, kBestDict, attrVals, attrValIndMap, testInsInd)
				if type(self.k) is int:
					if predY == testInsY:
						numCorrectCl += 1
				elif len(self.k) > 1:
					self.numInCl = []
					for p in range(len(self.k)):
						if predY[p] == testInsY:
							self.numInCl.append(0)
						else:
							self.numInCl.append(1)
				
			# Print result
			if type(self.k) is int:
				if lastAttName == 'response':
					testInsY = float(testInsY)
					MAE += abs(testInsY-predY)
					print 'Predicted value : '+ '%.6f'%predY + '	Actual value : '+ '%.6f'%testInsY
				elif lastAttName == 'class': 
					print 'Predicted class : '+ str(predY)+'	Actual class : '+ str(testInsY)
	
	
		if type(self.k) is int:
			if lastAttName == 'response':
				MAE = MAE/len(self.test_set.instances)
				print 'Mean absolute error : ' + '%.15f'%MAE # CHECK ABOUT THIS 15 DIGITS
				sys.stdout.write('Total number of instances : '+ str(len(self.test_set.instances)))
				sys.stdout.flush()
			elif lastAttName == 'class':
				print 'Number of correctly classified instances : '+ str(numCorrectCl)
				print 'Total number of instances : '+ str(len(self.test_set.instances))
	
				accuracy = float(numCorrectCl)/len(self.test_set.instances)
				sys.stdout.write('Accuracy : '+'%.16f'%accuracy)
				sys.stdout.flush()
		
	
	# Method for classification
	def classify(self,kBestDists,kBestDict, attrVals, attrValIndMap, testInsInd):
		
		if type(self.k) is int:
			yvalcount = []
			for i in range(len(attrVals)):
				yvalcount.append(0)
			
		elif len(self.k) > 1:
			
			yvalcount = []
			for i in range(len(self.k)):
				yvalcount.append([])
				
			for i in range(len(attrVals)):
				for j in range(len(self.k)):
					yvalcount[j].append(0)
			

		
		kBestDists.sort()
		numNNConsidered = 0
		done = False
		sum = 0
		
		for i in range(len(kBestDists)):
			dist = kBestDists[i]	
			d = collections.OrderedDict(sorted(kBestDict[dist].items()))
			for k1,v1 in d.iteritems():
	
				numNNConsidered += 1
				if type(self.k) is int:
					yvalcount[attrValIndMap[v1]] += 1
				elif len(self.k) > 1:
					if 1 <= numNNConsidered <= self.k[0]:
						yvalcount[0][attrValIndMap[v1]] += 1
						yvalcount[1][attrValIndMap[v1]] += 1
						yvalcount[2][attrValIndMap[v1]] += 1
					elif  self.k[0] < numNNConsidered <= self.k[1]:
						yvalcount[1][attrValIndMap[v1]] += 1
						yvalcount[2][attrValIndMap[v1]] += 1
					elif numNNConsidered > self.k[1]:
						yvalcount[2][attrValIndMap[v1]] += 1
						
				

				if type(self.k) is not int and len(self.k) > 1 and numNNConsidered == self.k[2]:
					done = True
					break

				if type(self.k) is int and numNNConsidered == self.k:
					done = True
					break	
					
			if done:
				break
				
		
		if type(self.k) is int:
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

					
		elif len(self.k)>1:
			predY = []
			# Finding class label
			for j in range(len(self.k)):
				maxVote = max(yvalcount[j])
				found = False
				unique = True
		
				for i in range(len(yvalcount[j])):
					if found  and yvalcount[j][i] == maxVote:
						unique = False
						break
					if not found and yvalcount[j][i] == maxVote:
						found = True
						firstLblInd = i
				
				if unique:		
					predY.append(attrVals[yvalcount[j].index(maxVote)])
				else:	
					predY.append(attrVals[firstLblInd])


		return predY
		
		
	
	# Method for regression 
	def regress(self,kBestDists,kBestDict):

		kBestDists.sort()
		numNNConsidered = 0
		done = False
		
		if type(self.k) is int:
			sum = 0
		elif len(self.k) > 1:
			sum = [0,0,0]
		
		
		for i in range(len(kBestDists)):
			dist = kBestDists[i]	
			d = collections.OrderedDict(sorted(kBestDict[dist].items()))
			for k1,v1 in d.iteritems():
				
				numNNConsidered += 1
				if type(self.k) is int:
					sum += float(v1)
				elif len(self.k) > 1:
					if 1 <= numNNConsidered <= self.k[0]:
						sum[0] += float(v1)
						sum[1] += float(v1)
						sum[2] += float(v1)
					elif  self.k[0] < numNNConsidered <= self.k[1]:
						sum[1] += float(v1)
						sum[2] += float(v1)
					elif numNNConsidered > self.k[1]:
						sum[2] += float(v1)						
				
				
				if type(self.k) is not int and len(self.k) > 1 and numNNConsidered == self.k[2]:
					done = True
					break
					
				
				if type(self.k) is int and numNNConsidered == self.k:
					done = True
					break		
					
			if done:
				break
		
		if type(self.k) is int:
			predY = sum/self.k
		elif len(self.k) > 1:
			predY = []
			for i in range(len(sum)):
				predY.append(sum[i]/self.k[i])
				
			
		
		return predY
		


		
	# Computing Euclidean distance squared
	def computeDist(self, feavec1,feavec2):
		dist = 0
		for i in range(len(feavec1)):
			dist +=(feavec1[i]-feavec2[i])**2
		dist = dist**(0.5)	
		return dist
		
		
	
		
		
class kNNselect(object):
	def __init__(self):
		self.train_set = None
		self.test_set = None
		self.k1 = None
		self.k2 = None
		self.k3 = None
		# Obtaining command line arguments
		self.obtainCLIArgs()
		kvec = [self.k1, self.k2, self.k3]
		
		
		kvecSorted = kvec
		kvecSorted.sort()
		
		kIndInSortOrder = {}
		for i in range(len(kvec)):
			kIndInSortOrder[kvec[i]] = kvecSorted.index(kvec[i])
		
		
		cv = leaveOneOut(self.train_set,kvecSorted)
		
		minIC = min(cv.numIncorrectCl)
		
		
		numTrainAtt = len(self.train_set.attributes)
		for k,v in self.train_set.attributes[numTrainAtt-1].iteritems():
			lastAttName = k
			attrVals = v
		

		if lastAttName == 'class':
			bestk = int(kvecSorted[cv.numIncorrectCl.index(minIC)])
			print 'Number of incorrectly classified instances for k = '+ str(self.k1) +' : '+ str(cv.numIncorrectCl[kIndInSortOrder[self.k1]])
			print 'Number of incorrectly classified instances for k = '+ str(self.k2) +' : '+ str(cv.numIncorrectCl[kIndInSortOrder[self.k2]])
			print 'Number of incorrectly classified instances for k = '+ str(self.k3) +' : '+ str(cv.numIncorrectCl[kIndInSortOrder[self.k3]])
			print 'Best k value : '+ str(bestk)
		elif lastAttName == 'response':
			bestk = int(kvecSorted[cv.numIncorrectCl.index(minIC)])
			print 'Mean absolute error for k = '+ str(self.k1) +' : '+ '%.16f'%(cv.numIncorrectCl[kIndInSortOrder[self.k1]]/len(self.train_set.instances))
			print 'Mean absolute error for k = '+ str(self.k2) +' : '+ '%.16f'%(cv.numIncorrectCl[kIndInSortOrder[self.k2]]/len(self.train_set.instances))
			print 'Mean absolute error for k = '+ str(self.k3) +' : '+ '%.16f'%(cv.numIncorrectCl[kIndInSortOrder[self.k3]]/len(self.train_set.instances))
			print 'Best k value : '+ str(bestk)
			
		bextKNN = kNN(self.train_set, self.test_set, bestk)
			
	# Method to obtain data from command line arguments
	def obtainCLIArgs(self):

		trainFileName = sys.argv[1]

		testFileName = sys.argv[2]

		self.k1 = int(sys.argv[3])
		self.k2 = int(sys.argv[4])
		self.k3 = int(sys.argv[5])
		self.train_set = getContents(trainFileName)  
		self.test_set = getContents(testFileName)
		
			
class dataSubset(object):
	def __init__(self):
		self.attributes = None
		self.instances = None
	
			
class leaveOneOut(object):
	def __init__(self,train_set,kvecSorted):
		
		self.numIncorrectCl = [0,0,0]
		
		self.leaveOneOutCV(train_set,kvecSorted)
		
		
	def leaveOneOutCV(self, train_set, kvecSorted):
		
		for trainInd in range(len(train_set.instances)):
			trainSetCv = dataSubset()
			trainSetCv.attributes = train_set.attributes
			trainSetCv.instances = train_set.instances[0:trainInd] + train_set.instances[trainInd+1:]
			
			
			testSetCv =  dataSubset()
			testSetCv.attributes = train_set.attributes
			testSetCv.instances = [train_set.instances[trainInd]]
		
			kNNTrainInst = kNN(trainSetCv, testSetCv, kvecSorted)
			for h in range(len(self.numIncorrectCl)):
				self.numIncorrectCl[h] += kNNTrainInst.numInCl[h]
		
		

		
ks = kNNselect()		
