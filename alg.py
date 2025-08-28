import math
import random
import time

import numpy as np

def ReleaseGuaranteeAlg(
		N: int,
		M: int,
		Q: int,
		KList: np.ndarray,
		AList: np.ndarray,
		aList: np.ndarray,
		gammaList: np.ndarray,
		rhoList: np.ndarray,
		consumptionArray: np.ndarray,
		valueArray: np.ndarray,
		granularity: float,
		consumptionTrace: list,
    WordCountBin: list,
) -> [float, float] :
	# initialization
	thetaList = np.zeros(M)
	phiList = np.zeros(Q)
	WList = np.zeros(M)
	SList = np.zeros(Q)
	uList = np.zeros(M)
	vList = np.zeros(Q)
	hList = np.array([[0 for w in range(int(1/granularity)+1)] for m in range(M)])
	fList = np.array([[0 for s in range(int(1 / granularity) + 1)] for q in range(Q)])
	accumulatedReward = 0
	averageRunTime = 0
	decisionList = []
	for n in range(N):
		startTime = time.time()
		# Decision-Making
		process = True
		for m in range(M):
			if WList[m] <= thetaList[m] and (KList[m]-WList[m])>=1:
				uList[m]=1
			else:
				# assert ((KList[m]-WList[m])>=1)
				uList[m]=0
				process = False
				break
		for q in range(Q):
			if SList[q]<=phiList[q] and (AList[q]-SList[q])>=1:
				vList[q]=1
			else:
				vList[q]=0
				process = False
				break
		# Execution
		if process:
			decisionList.append(n)
			eList = [consumptionTrace[i][n] for i in range(M)]
			WList = WList+eList
			for q in range(Q):
				SList[q] = SList[q] + np.sum(aList[q]*eList)
			accumulatedReward+=valueArray[n]
		# Evaluation
		# 1. h function
		hList = update_hList(
			granularity=granularity,
			thetaList=thetaList,
			hList=hList,
			gList=consumptionArray,
			n=n,
			bin = int(WordCountBin[n]),
			M=M,
			gammaList=gammaList,
		)
		# 2. f function
		fList = update_fList(
			granularity=granularity,
			phiList=phiList,
			fList=fList,
			gList=consumptionArray,
			aList=aList,
			M=M,
			Q=Q,
			n=n,
			bin=int(WordCountBin[n]),
			rhoList=rhoList
		)
		# thetaList
		for m in range(M):
			theta = 1
			threshold = 0
			for w in range(len(hList[m])):
				threshold += hList[m][w]
				if threshold>=gammaList[m]:
					theta = w*granularity
					break
			thetaList[m]=theta
		# phiList
		for q in range(Q):
			phi = 1
			threshold = 0
			for s in range(len(fList[q])):
				threshold += fList[q][s]
				if threshold>=rhoList[q]:
					phi = s*granularity
					break
			phiList[q]=phi
		endTime = time.time()
		averageRunTime += (endTime-startTime)
	averageRunTime /= N
	return accumulatedReward, averageRunTime

def update_fList(
		granularity: float,
		phiList: np.ndarray,
		fList: np.ndarray,
		gList: np.ndarray,
		aList: np.ndarray,
		M: int,
		Q: int,
		n: int,
		bin: int,
		rhoList: np.ndarray
) -> np.ndarray:
	# calculate gBar
	gBarList = calculate_gBar(
			gList=gList,
			aList=aList,
			M=M,
			Q=Q,
			n=n,
			bin=bin,
		granularity=granularity
	)
	if n== 0:
		gPlusList = np.array([[1 - rhoList[q]] + [0 for i in range(len(gBarList[q]) - 1)] for q in range(Q)])
		return gBarList+gPlusList
	else:
		fListNew = []
		for q in range(Q):
			appro_phi = int(phiList[q] / granularity)
			fBar = np.concatenate((np.ones(appro_phi + 1), np.zeros(len(fList[q].tolist()) - (appro_phi + 1)))) * fList[q]
			term1 = np.convolve(fBar, gBarList[q], mode='full')
			fNew = (term1 + np.concatenate((fList[q] - fBar, np.zeros(len(term1) - len(fBar)))))
			if np.sum(fNew) > 0:
				fNew /= np.sum(fNew)
			fListNew.append(fNew.tolist())
		return np.array(fListNew)


def calculate_gBar(
		gList: np.ndarray,
		aList: np.ndarray,
		M: int,
		Q: int,
		n: int,
		bin: int,
		granularity: float
) -> np.ndarray:
	gBarList = []
	for q in range(Q):
		scaled_gList = []
		for m in range(M):
			scaled_g = [gList[m, bin, int(s / aList[q,m])] for s in
									range(int((1 * aList[q,m]) / granularity) + 1)]
			if sum(scaled_g)<=0.1:
				scaled_g[0]=1
			scaled_gList.append((np.array(scaled_g)/sum(scaled_g)).tolist())
		gBar = np.array(scaled_gList[0])
		for m in range(1,M):
			gBar = np.convolve(gBar,np.array(scaled_gList[m]), mode='full')
		gBarScale = gBar

		difference = round(1/granularity + 1 - len(gBarScale))
		gBarScale = np.concatenate((np.zeros(difference),gBar))
		if np.sum(gBarScale)>0:
			gBarScale/=np.sum(gBarScale)
		assert len(gBarScale)==1/granularity + 1
		gBarList.append(gBarScale)
	return np.array(gBarList)


def update_hList(
		granularity: float,
		thetaList: np.ndarray,
		hList: np.ndarray,
		gList: np.ndarray,
		n: int,
		bin: int,
		M: int,
		gammaList: np.ndarray
) -> np.ndarray:
	if n == 0:
		gPlusList = np.array([[1-gammaList[m]]+[0 for i in range(len(gList[m,bin])-1)] for m in range(M)])
		return np.array([(gList[m,bin]+gPlusList[m]).tolist() for m in range(M)])
	else:
		hListNew = []
		for m in range(M):
			appro_theta = int(thetaList[m]/granularity)
			hBar = np.concatenate((np.ones(appro_theta+1), np.zeros(len(hList[m].tolist())-(appro_theta+1))))*hList[m]
			term1 = np.convolve(hBar,gList[m,bin], mode='full')
			hNew = term1+np.concatenate((hList[m]-hBar,np.zeros(len(term1)-len(hBar))))
			if np.sum(hNew) > 0:
				hNew /= np.sum(hNew)
			hListNew.append(hNew.tolist())
		return np.array(hListNew)

def sample_eList(
		M: int,
		consumptionArray: np.ndarray,
		n: int
) -> np.ndarray:
	eList = []
	for m in range(M):
		dice = random.random()
		if dice>=np.sum(consumptionArray[m][n]):
			eList.append(1)
			continue
		else:
			e = 0
			for w in range(len(consumptionArray[m][n])):
				if dice<=consumptionArray[m][n][w]:
					e = w*(1/(len(consumptionArray[m][n])-1))
					break
				else:
					dice = dice - consumptionArray[m][n][w]
			eList.append(e)
	return np.array(eList)

def generateConsumptionArray(	# M*N
		N: int,
		M: int,
		BList: np.ndarray,
		granularity: float
) -> np.ndarray:
	averageConsumptionRaw = [[random.random() for _ in range(N)] for _ in range(M)]
	total = [sum(averageConsumptionRaw[m]) for m in range(M)]
	averageConsumption = [[averageConsumptionRaw[m][n] * BList[m] / total[m] for n in range(N)] for m in range(M)]
	std = 0.1
	PDFLength = int(1/granularity)
	consumptionPDF = [[] for m in range(M)]
	pdfScale = np.arange(0, 1 + granularity, granularity)
	for m in range(M):
		for n in range(N):
			PrList: np.ndarray = 1 / (np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((pdfScale - averageConsumption[m][n]) / std) ** 2)
			PrList[0] = 0
			PrList /= np.sum(PrList)
			error = 1
			while abs(error)>0.5:
				error = averageConsumption[m][n]-sum(PrList*pdfScale)
				PrList[int(0.5/granularity)+1] = PrList[int(0.5/granularity)+1]-0.001*error
				PrList /= np.sum(PrList)
			prList = PrList.tolist()
			consumptionPDF[m].append(prList)
	return np.array(consumptionPDF)

def generateConsumptionArrayBeta(
		N: int,
		M: int,
		BList: np.ndarray,
		granularity: float,
) -> np.ndarray:
	averageConsumptionRaw = [[np.random.random() for _ in range(N)] for _ in range(M)]
	total = [sum(averageConsumptionRaw[m]) for m in range(M)]
	averageConsumption = [
		[averageConsumptionRaw[m][n] * BList[m] / total[m] for n in range(N)] for m in range(M)
	]

	PDFLength = int(1 / granularity)
	consumptionPDF = [[] for m in range(M)]
	pdfScale = np.arange(0, 1 + granularity, granularity)

	for m in range(M):
		for n in range(N):
			PrList = np.zeros(round(1/granularity) +1)
			if averageConsumption[m][n]>0.5:
				PrList[round(1/granularity)] = (averageConsumption[m][n]-0.5)/0.5
				PrList[round(0.5/granularity)] = 1-PrList[round(1/granularity)]
			elif averageConsumption[m][n]>=0.1:
				PrList[round(0.5 / granularity)] = (averageConsumption[m][n]-0.1)/0.4
				PrList[round(0.1 / granularity)] = 1-PrList[round(0.5 / granularity)]
			else:
				PrList[0]=1

			PrList /= np.sum(PrList)
			error = 1
			consumptionPDF[m].append(PrList.tolist())

	return np.array(consumptionPDF)

def generateValueArray(
		N: int,
		maxV: int,
		minV: int,
		granularity: float
) -> np.ndarray:
	valueArray = np.zeros(N)
	valueCandidates = np.arange(minV, maxV + granularity, granularity).tolist()
	for n in range(N):
		valueArray[n] = random.choice(valueCandidates)
	return valueArray

def generate_aList(
		M: int,
		Q: int,
		etaList: np.ndarray,
		BList: np.ndarray
) -> [np.ndarray,np.ndarray]:
	aList = [[] for q in range(Q)]
	for q in range(Q):
			a = np.array([random.random() for m in range(M)])
			a /= (np.sum(a))
			a = a.tolist()
			aList[q]=a
	AList = [np.sum(np.array(aList[q])*BList)/etaList[q] for q in range(Q)]
	return np.array(aList), np.array(AList)

# generate a_{n,m} for CR validation experiment
def generate_aListFromA(
		M: int,
		Q: int,
		etaList: np.ndarray,
		KList: np.ndarray
) ->np.ndarray:
	aList = [[] for q in range(Q)]
	for q in range(Q):
		a = np.array([random.random() for m in range(M)])
		a /= (np.sum(a))
		a = a.tolist()
		aList[q]=a
	return np.array(aList)

