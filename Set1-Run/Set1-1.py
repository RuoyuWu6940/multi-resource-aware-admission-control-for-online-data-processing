import alg
import numpy as np

for offsetK in [0,100,200,300,400,500]:

    N = 1000    # Number of tasks
    M = 3   #   Number of individual resource
    Q = 1   # Number of joint resource
    minV = 1    # min reward
    maxV = 50   # max reward
    granularity = 0.1   # granularity for resource consumption
    K =200+offsetK  # individual resource budget
    eta = 1.05  # eta
    KList = K*np.ones(M)
    betaList = 1*np.ones(M)
    etaList = eta*np.ones(Q)

    A = K/eta   # joint resource budget
    AList = A*np.ones(Q)
    BList = KList * betaList    # still the individual resource budget
    gammaList = (np.ones(M)/betaList)*(1-np.power(KList, -1/2)) # gamma_m for each individual resource m
    gammaList = np.clip(gammaList, None, 1)
    rhoList = (np.ones(Q)/etaList)*(1-np.power(AList, -1/2))    # rho_n for each joint resource n
    rhoList = np.clip(rhoList, None, 1)
    for m in range(M):
        if BList[m]/N>=1:
            print("Average Consumption ", BList[m]/N, " of resource ", m, " is greater than 1.")
            assert BList[m]/N<1
    Traces = 10 # 10 sample paths
    Rounds = 15 # 15 red dots
    praticalCRs = []    # red dots (i.e., performance ratio)
    for j in range(Rounds):
        aList = alg.generate_aListFromA(    # generate a_{n,m}
            M=M,
            Q=Q,
            etaList=etaList,
            KList=KList
        )
        consumptionArray = alg.generateConsumptionArray(    # generate resource consumption for tasks
            N=N,
            M=M,
            BList=BList,
            granularity=granularity
        )
        valueArray = alg.generateValueArray(    # generate reward for tasks
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
    print("K: ", K)
    print("practical CR: ", praticalCRs)