import math
import random
import time
import numpy as np
import alg


# Greedy
def greedyAlg(
	N: int,
	M: int,
	Q: int,
	KList: np.ndarray,
	AList: np.ndarray,
	aList: np.ndarray,
	consumptionTrace: list,
	valueArray: np.ndarray,
) -> [float, float]:
	# initialization
	WList = np.zeros(M)
	SList = np.zeros(Q)
	accumulatedReward = 0
	averageRunTime = 0
	decisionList = []
	for n in range(N):
		startTime = time.time()
		# Decision-Making
		process = True
		for m in range(M):
			if (KList[m] - WList[m]) < 1:
				process = False
				break
		for q in range(Q):
			if (AList[q] - SList[q]) < 1:
				process = False
				break
		# Execution
		if process:
			decisionList.append(n)
			eList = [consumptionTrace[i][n] for i in range(M)]
			WList = WList + eList
			for q in range(Q):
				SList[q] = SList[q] + np.sum(aList[q] * eList)
			accumulatedReward += valueArray[n]
		endTime = time.time()
		averageRunTime += (endTime - startTime)
	averageRunTime /= N
	# print("greedyDecision:", decisionList)
	return accumulatedReward, averageRunTime

def adaptive(
		N: int,
		M: int,
		Q: int,
		KList: np.ndarray,
		AList: np.ndarray,
		aList: np.ndarray,
		etaList: np.ndarray,
		consumptionTrace: list,
		valueArray: np.ndarray,
) -> [float, float]:
	# initialization
	WList = np.zeros(M)
	SList = np.zeros(Q)
	accumulatedReward = 0
	averageRunTime = 0
	eta = np.max(etaList)
	for n in range(N):
		startTime = time.time()
		processed = 0
		# Decision-Making
		process = True
		for m in range(M):
			if (KList[m] - WList[m]) < 1:
				process = False
				break
		for q in range(Q):
			if (AList[q] - SList[q]) < 1:
				process = False
				break
		if n>0:
			if processed/n>1/eta:
				process=False
		# Execution
		if process:
			processed+=1
			eList = [consumptionTrace[i][n] for i in range(M)]
			WList = WList + eList
			for q in range(Q):
				SList[q] = SList[q] + np.sum(aList[q] * eList)
			accumulatedReward += valueArray[n]
		endTime = time.time()
		averageRunTime += (endTime - startTime)
	averageRunTime /= N
	return accumulatedReward, averageRunTime

# Random
def safeRandomEta(
		N: int,
		M: int,
		Q: int,
		KList: np.ndarray,
		AList: np.ndarray,
		aList: np.ndarray,
		etaList: np.ndarray,
		consumptionTrace: list,
		valueArray: np.ndarray,
		betaList: np.ndarray
) -> [float, float]:
	# initialization
	WList = np.zeros(M)
	SList = np.zeros(Q)
	accumulatedReward = 0
	averageRunTime = 0
	if Q>0:
		threshold = min(1,np.max(etaList))
	else:
		threshold = 1
	for n in range(N):
		startTime = time.time()
		# Decision-Making
		process = False
		dice = random.random()
		if dice <= threshold:
			process = True
		for m in range(M):
			# print(m,WList[m], thetaList[m], KList[m])
			if (KList[m]-WList[m])<1:
				process = False
				break
		for q in range(Q):
			if (AList[q]-SList[q])<1:
				process = False
				break
		#Execution
		if process:
			eList = [consumptionTrace[i][n] for i in range(M)]
			WList = WList+eList
			for q in range(Q):
				SList[q] = SList[q] + np.sum(aList[q]*eList)
			accumulatedReward+=valueArray[n]
		endTime = time.time()
		averageRunTime += (endTime - startTime)
	averageRunTime /= N
	return accumulatedReward, averageRunTime

# OSMA
def safeSingleK(
		N: int,
		M: int,
		Q: int,
		KList: np.ndarray,
		AList: np.ndarray,
		aList: np.ndarray,
		consumptionArray: np.ndarray,
		consumptionTrace: list,
		valueArray: np.ndarray,
		granularity: float,
		betaList: np.ndarray,
		WordCountBin: list
) -> [float, float]:
	# initialization
	m_K = 0
	K = np.max(KList)
	for m in range(M):
		if KList[m]<=K:
			m_K = m
			K = KList[m]
	gamma = (1/betaList[m_K])*(1-1/(math.pow(K+3, 1/2)))

	theta = 0
	WList = np.zeros(M)
	SList = np.zeros(Q)
	hList = np.array([[0 for w in range(int(1/granularity)+1)] for m in range(1)])
	accumulatedReward = 0
	averageRunTime = 0
	for n in range(N):
		startTime = time.time()
		# Decision-Making
		process = False
		if WList[m_K] <= theta and (K-WList[m_K])>=1:
			process = True
			for m in range(M):
				if (KList[m]-WList[m])<1:
					process = False
					break
			for q in range(Q):
				if (AList[q]-SList[q])<1:
					process = False
					break
		# Execution
		if process:
			eList = eList = [consumptionTrace[i][n] for i in range(M)]
			WList = WList+eList
			for q in range(Q):
				SList[q] = SList[q] + np.sum(aList[q]*eList)
			accumulatedReward+=valueArray[n]
		# Evaluation
		# 1. h function
		hList = alg.update_hList(
			granularity=granularity,
			thetaList=np.array([theta]),
			hList=hList,
			gList=np.array([consumptionArray[m_K].tolist()]),
			n=n,
			bin=int(WordCountBin[n]),
			M=1,
			gammaList=np.array([gamma])
		)
		# thetaList
		for m in range(1):
			thetaNew = 1
			threshold = 0
			for w in range(len(hList[m])):
				threshold += hList[m][w]
				if threshold>=gamma:
					thetaNew = w*granularity
					break
			theta=thetaNew
		endTime = time.time()
		averageRunTime += (endTime-startTime)
	# print("SIngleK Decision:",decisionList)
	# print("thetaList", thetaList)
	# print("WList",WList)
	averageRunTime /= N
	return accumulatedReward, averageRunTime

