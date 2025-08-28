import alg
import numpy as np
for q in [1,2,3,4,5]:

    N = 1000
    M = 5
    Q = q
    minV = 1
    maxV = 50
    granularity = 0.1
    K =800
    eta = 1.05
    KList = K*np.ones(M)
    betaList = 1*np.ones(M)
    etaList = eta*np.ones(Q)
    betaList = betaList
    etaList = etaList
    A = K/eta
    AList = A*np.ones(Q)
    BList = KList * betaList
    gammaList = (np.ones(M)/betaList)*(1-np.power(KList, -1/2))
    gammaList = np.clip(gammaList, None, 1)
    rhoList = (np.ones(Q)/etaList)*(1-np.power(AList, -1/2))
    rhoList = np.clip(rhoList, None, 1)
    for m in range(M):
        if BList[m]/N>=1:
            print("Average Consumption ", BList[m]/N, " of resource ", m, " is greater than 1.")
            assert BList[m]/N<1
    Traces = 10
    Rounds = 15
    praticalCRs = []
    for j in range(Rounds):
        aList = alg.generate_aListFromA(
            M=M,
            Q=Q,
            etaList=etaList,
            KList=KList
        )
        consumptionArray = alg.generateConsumptionArray(
            N=N,
            M=M,
            BList=BList,
            granularity=granularity
        )
        valueArray = alg.generateValueArray(
            N=N,
            maxV=maxV,
            minV=minV,
            granularity=granularity
        )
        print("aList: ", aList)
        print("N: ", N, "; M: ",M, "; Q: ", Q, "; range of value: [", minV, ",", maxV, "]")
        print("KList: ", KList)
        print("AList: ", AList)
        print("etaList: ", etaList)
        print("prod_beta_eta(C): ", np.prod(1/etaList))

        accumulatedReward = 0
        averageRunTime = 0
        for i in range(Traces):
            thisAccumulatedReward, thisAverageRunTime = alg.ReleaseGuaranteeAlg(
                N=N,
                M=M,
                Q=Q,
                KList=KList,
                AList=AList,
                aList=aList,
                gammaList=gammaList,
                rhoList=rhoList,
                consumptionArray=consumptionArray,
                valueArray=valueArray,
                granularity=granularity
            )
            accumulatedReward+=thisAccumulatedReward
            averageRunTime += thisAverageRunTime
        expectedReward = accumulatedReward/Traces
        averageRunTime = averageRunTime/Traces
        print("averageRunTime(per task)(ms): ", averageRunTime*1000)
        print("practical CR: ", expectedReward/np.sum(valueArray))
        praticalCRs.append(expectedReward/np.sum(valueArray))
    theoreticalCR = np.prod(gammaList)*np.prod(rhoList)
    print("theoretical CR: ", theoreticalCR)
    print("prod_beta_eta(C): ", np.prod(1/etaList))
    print("Q: ", Q)
    print("practical CR: ", praticalCRs)