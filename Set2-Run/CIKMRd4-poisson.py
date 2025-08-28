import numpy as np
import alg
import pandas as pd
import bennchmark as bmk

def traceStatisticSetUp():
    individualBudgetFactor = 1
    jointBudgetFactor = 0.8
    BestTraceGuess = 1
    numIndi = 3
    # numIndi is the number of individual resource. numIndi=2 is CPU+GPU; numIndi=3 is CPU+GPU+energy
    numJoint = 1
    # numJoint is the number of joint resource. numJoint=0 does not consider money; numJoint=1 consider money

    traceLengthRange = [10,20]
    caseInLength = \
        [10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20]
    caseNumberPerLength = [caseInLength.count(i) for i in range(traceLengthRange[0],traceLengthRange[1]+1)]
    caseStartingNumber = 0

    RoundsForAverage=50

    ## prepare trace data
    expectedCPUTime = 0.0050445705
    expectedGPUTime = 73.761950765
    expectedJoule = 8113.9348103955
    expectedIndiRes = [expectedCPUTime, expectedGPUTime, expectedJoule]

    maxSingleCPU = 0.002154
    maxSingleGPU = 18.67264
    maxSingleJoule = 1834.01648
    maxSingleIndiRes = [maxSingleCPU, maxSingleGPU, maxSingleJoule]

    indiResTableLable = ['total_cpu_time_sec', 'time_inference_sec','energy_total']

    CPUmoney = 0.005  # cent per CPU-second
    GPUmoney = 0.02  # cent per CPU-second
    JouleMoney = 1.16e-6  # cent per joule
    indiToJointCoefficiant = [0.005, 0.02, 1.16e-6]

    cpuComsumptionArray= \
        [[0, 0.0, 0.0, 0.4694835680751174, 0.48708920187793425, 0.03755868544600939, 0.0035211267605633804, 0.0,
          0.0011737089201877935, 0.0, 0.0011737089201877935],
         [0, 0.0, 0.0, 0.3484848484848485, 0.601010101010101, 0.03787878787878788, 0.006313131313131313,
          0.0025252525252525255, 0.0, 0.0012626262626262627, 0.0025252525252525255],
         [0, 0.0, 0.0, 0.1997533908754624, 0.689272503082614, 0.09864364981504316, 0.006165228113440197,
          0.0012330456226880395, 0.0012330456226880395, 0.0, 0.0036991368680641184],
         [0, 0.0, 0.0, 0.07402597402597402, 0.7467532467532467, 0.15064935064935064, 0.01818181818181818,
          0.003896103896103896, 0.0, 0.0012987012987012987, 0.005194805194805195],
         [0, 0.0, 0.0, 0.00904392764857881, 0.5658914728682171, 0.35012919896640826, 0.059431524547803614,
          0.011627906976744186, 0.0, 0.0, 0.003875968992248062]]
    gpuConsumptionArray= \
        [[0, 0.0, 0.0, 0.4694835680751174, 0.48708920187793425, 0.03755868544600939, 0.0035211267605633804, 0.0,
          0.0011737089201877935, 0.0, 0.0011737089201877935],
         [0, 0.0, 0.0, 0.3484848484848485, 0.601010101010101, 0.03787878787878788, 0.006313131313131313,
          0.0025252525252525255, 0.0, 0.0012626262626262627, 0.0025252525252525255],
         [0, 0.0, 0.0, 0.1997533908754624, 0.689272503082614, 0.09864364981504316, 0.006165228113440197,
          0.0012330456226880395, 0.0012330456226880395, 0.0, 0.0036991368680641184],
         [0, 0.0, 0.0, 0.07402597402597402, 0.7467532467532467, 0.15064935064935064, 0.01818181818181818,
          0.003896103896103896, 0.0, 0.0012987012987012987, 0.005194805194805195],
         [0, 0.0, 0.0, 0.00904392764857881, 0.5658914728682171, 0.35012919896640826, 0.059431524547803614,
          0.011627906976744186, 0.0, 0.0, 0.003875968992248062]]
    jouleConsumptionArray = \
        [[0, 0.0, 0.0011737089201877935, 0.0539906103286385, 0.23591549295774647, 0.27699530516431925, 0.20539906103286384, 0.12441314553990611, 0.05985915492957746, 0.04225352112676056, 0.0], [0, 0.0, 0.0, 0.017676767676767676, 0.13383838383838384, 0.29924242424242425, 0.24621212121212122, 0.14267676767676768, 0.08207070707070707, 0.07828282828282829, 0.0], [0, 0.0, 0.0, 0.0036991368680641184, 0.093711467324291, 0.18372379778051787, 0.24907521578298397, 0.20468557336621454, 0.12946979038224415, 0.13563501849568435, 0.0], [0, 0.0, 0.0, 0.005194805194805195, 0.04285714285714286, 0.14155844155844155, 0.2519480519480519, 0.22467532467532467, 0.14935064935064934, 0.18441558441558442, 0.0], [0, 0.0, 0.0, 0.002583979328165375, 0.015503875968992248, 0.12790697674418605, 0.1731266149870801, 0.16925064599483206, 0.16666666666666666, 0.3449612403100775, 0.0]]
    consumptionArray=np.array([cpuComsumptionArray,gpuConsumptionArray,jouleConsumptionArray])
    return [individualBudgetFactor, jointBudgetFactor,BestTraceGuess,expectedIndiRes, consumptionArray, traceLengthRange, caseNumberPerLength, caseStartingNumber, RoundsForAverage,numIndi,numJoint, maxSingleIndiRes, indiResTableLable, indiToJointCoefficiant]

def index_in_df_sp(csv_file: str, query1: int, query2 : int):
    '''
    query1: 输入需要session中含有多少个task 5-20
    query2: 输入这个session的编号，比如0-5，前面的数可以多一点
    如果不想query哪个列就输入None
    e.g.
    index_in_df_sp("cikm_data.csv", 5, None)
    index_in_df_sp("cikm_data.csv", None, 5)
    index_in_df_sp("cikm_data.csv", None, None)
    '''
    df = pd.read_csv(csv_file)
    if query1 == None and query2 == None:
        return
    elif query1 == None:
        return df[df["query2"] == query2].to_csv("cikm_data_query.csv", index=False)
    elif query2 == None:
        return df[df["query1"] == query1].to_csv("cikm_data_query.csv", index=False)
    return df[(df["query1"] == query1) & (df["query2"] == query2)].to_csv("cikm_data_query.csv", index=False)

if __name__ == '__main__':
    (individualBudgetFactor,
     jointBudgetFactor,
     BestTraceGuess,
     expectedIndiRes,
     consumptionArray,
     traceLengthRange,
     caseNumberPerLength,
     caseStartingNumber,
     RoundsForAverage,
     numIndi,
     numJoint,
     maxSingleIndiRes,
     indiResTableLable,
     indiToJointCoefficiant) = traceStatisticSetUp()



    ## Check caseNumberPerLength
    if len(caseNumberPerLength) != traceLengthRange[1]-traceLengthRange[0]+1:
        print("caseNumberPerLength", caseNumberPerLength, "does not equal traceLengthRange", traceLengthRange)
        exit()

    ## Figure Data
    # Average on all Trace
    TraceAverageOurs = 0
    TraceAverageGreedy = 0
    TraceAverageSingleK = 0
    TraceAverageRandomEta = 0
    TraceAverageAdaptive = 0

    AveragePerLengthListOurs = []
    AveragePerLengthListGreedy = []
    AveragePerLengthListSingleK = []
    AveragePerLengthListRandomEta = []
    AveragePerLengthListAdaptive = []

    RuntimePerPromptOurs = 0
    RuntimePerPromptGreedy = 0
    RuntimePerPromptSingleK = 0
    RuntimePerPromptRandomEta = 0
    RuntimePerPromptAdaptive = 0

    RuntimePerLengthListOurs = []
    RuntimePerLengthListGreedy = []
    RuntimePerLengthListSingleK = []
    RuntimePerLengthListRandomEta = []
    RuntimePerLengthListAdaptive = []
    ## Figure Data

    for traceLength in range(traceLengthRange[0],traceLengthRange[1]+1):

        print("Running sessions with length ", traceLength)

        ## Average Reward per Length
        AverageOverALengthOurs = 0
        AverageOverALengthGreedy = 0
        AverageOverALengthSingleK = 0
        AverageOverALengthRandomEta = 0
        AverageOverALengthAdaptive = 0

        ## Average Runtime per Length
        AverageRuntimeALengthOurs = 0
        AverageRuntimeALengthGreedy = 0
        AverageRuntimeALengthSingleK = 0
        AverageRuntimeALengthRandomEta = 0
        AverageRuntimeALengthAdaptive = 0

        for case in range(caseStartingNumber, caseNumberPerLength[traceLength-traceLengthRange[0]]+caseStartingNumber):

            DATA_DIR = "cikm_data_poisson.csv"
            index_in_df_sp(DATA_DIR, traceLength,case)
            taskTable = pd.read_csv("cikm_data_query.csv")
            taskTable["energy_total"] = taskTable["total_cpu_time_sec"] * 55 + taskTable['time_inference_sec'] * 110

            valueArray = taskTable['satisfaction_score'].to_numpy()  # trace's reward
            promptWordCountBin = taskTable['word_count_bin'].to_numpy()
            # consumptionTrace = [cpuB,gpuB,jouleB]
            consumptionTrace = [(taskTable[indiResTableLable[i]].to_numpy()/maxSingleIndiRes[i]).tolist() for i in range(numIndi)]
            print(len(consumptionTrace[i]) for i in range(numIndi)) # check number of requests
            EGList = np.array(consumptionTrace) # single resource consumption trace (normalized)
            BList = np.array([sum(consumptionTrace[i]) for i in range(numIndi)])    # overall consumption of single resource

            aList = np.array([[indiToJointCoefficiant[i]/sum(indiToJointCoefficiant[0:numIndi]) for i in range(numIndi)]])  # normalized coefficient
            EGAList = sum([EGList[i]*aList[0,i] for i in range(numIndi)]) # trace's energy consumption list; and money cost (normalized)
            print(np.max(EGAList[0]), np.min(EGAList[1]))  #maximum energy consumption and money cost
            N = len(valueArray) # task number
            assert N==traceLength
            M = numIndi   # individual resource
            Q = numJoint   # joint resource
            minV = 0    # min reward
            maxV = 1   # max reward

            granularity = 0.1  # consumption granularity
            KList = np.array(expectedIndiRes) # expected CPU, GPU usage
            betaList = np.array([individualBudgetFactor for i in range(M)])#1-2, \eta = 4/3 # individual resource scarcity factor
            etaList = np.array([jointBudgetFactor])	#3-4, \beta = 1   # joint resource scarcity factor
            if numJoint == 0:
                etaList = np.array([1])

            KList = np.array([KList[i]/maxSingleIndiRes[i] for i in range(numIndi)]) # (normalized single budget)
            AList = np.array([(sum([KList[i]*aList[0,i] for i in range(numIndi)])) * etaList[0]])  # Expected energy, money budget (normalized)

            gammaList = (np.ones(M) / betaList) * (1 - np.power(KList, -1 / 2))
            gammaList = np.clip(gammaList, None, 1)
            rhoList = np.ones(1)
            if Q>0:
                rhoList = (np.ones(Q) / etaList) * (1 - np.power(AList, -1 / 2))
                rhoList = np.clip(rhoList, None, 1)
            print("gammaList", gammaList)
            print("rhoList", rhoList)
            print("aList: ", aList)
            print("N: ", N, "; M: ", M, "; Q: ", Q, "; range of value: [", minV, ",", maxV, "]")
            print("KList: ", KList)
            print("betaList: ", betaList)
            print("AList: ", AList)
            print("etaList: ", etaList)
            print("prod_beta_eta: ", np.prod(1 / betaList) * np.prod(1 / etaList))
            print("prod_K_A: ", np.prod(1 - np.power(KList, -1 / 2)) * np.prod(1 - np.power(AList, -1 / 2)))

            print("value(satisfaction):", valueArray)
            print("consumptionTrace",consumptionTrace)
            ## prepare trace data complete

            ## Trace Run
            print("Case {} of length {}".format(case,traceLength))
            Rounds = 1

            accumulatedRewardOurs = 0
            accumulatedRewardGreedy = 0
            accumulatedRewardSingleK = 0
            accumulatedRewardSafeRandomEta = 0
            accumulatedRewardAdaptive = 0

            accumulatedRuntimeOurs = 0
            accumulatedRuntimeGreedy = 0
            accumulatedRuntimeSingleK = 0
            accumulatedRuntimeSafeRandomEta = 0
            accumulatedRuntimeAdaptive = 0

            for i in range(RoundsForAverage):
                # ours
                thisAccumulatedRewardOurs, thisAverageRunTimeOurs = alg.ReleaseGuaranteeAlg(
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
                    granularity=granularity,
                    consumptionTrace=consumptionTrace,
                    WordCountBin=promptWordCountBin
                )
                accumulatedRewardOurs += thisAccumulatedRewardOurs
                accumulatedRuntimeOurs += thisAverageRunTimeOurs


                thisAccumulatedRewardGreedy, thisAverageRunTimeGreedy = bmk.greedyAlg(
                    N=N,
                    M=M,
                    Q=Q,
                    KList=KList,
                    AList=AList,
                    aList=aList,
                    consumptionTrace=consumptionTrace,
                    valueArray=valueArray
                )
                accumulatedRewardGreedy += thisAccumulatedRewardGreedy
                accumulatedRuntimeGreedy += thisAverageRunTimeGreedy

                thisAccumulatedRewardSingleK, thisAverageRunTimeSingleK = bmk.safeSingleK(
                    N=N,
                    M=M,
                    Q=Q,
                    KList=KList,
                    AList=AList,
                    aList=aList,
                    consumptionArray=consumptionArray,
                    valueArray=valueArray,
                    granularity=granularity,
                    betaList=betaList,
                    consumptionTrace=consumptionTrace,
                    WordCountBin=promptWordCountBin,
                )
                accumulatedRewardSingleK += thisAccumulatedRewardSingleK
                accumulatedRuntimeSingleK += thisAverageRunTimeSingleK

                thisAccumulatedRewardSafeRandomEta, thisAverageRunTimeSafeRandomEta = bmk.safeRandomEta(
                    N=N,
                    M=M,
                    Q=Q,
                    KList=KList,
                    AList=AList,
                    aList=aList,
                    consumptionTrace=consumptionTrace,
                    valueArray=valueArray,
                    betaList=betaList,
                    etaList=etaList
                )
                accumulatedRewardSafeRandomEta += thisAccumulatedRewardSafeRandomEta
                accumulatedRuntimeSafeRandomEta += thisAverageRunTimeSafeRandomEta

                thisAccumulatedRewardAdaptive, thisAverageRunTimeAdaptive = bmk.adaptive(
                    N=N,
                    M=M,
                    Q=Q,
                    KList=KList,
                    AList=AList,
                    aList=aList,
                    consumptionTrace=consumptionTrace,
                    valueArray=valueArray,
                    etaList=etaList
                )
                accumulatedRewardAdaptive += thisAccumulatedRewardAdaptive
                accumulatedRuntimeAdaptive += thisAverageRunTimeAdaptive

            ## Average Reward on a case
            expectedRewardOurs = accumulatedRewardOurs / RoundsForAverage
            expectedRewardGreedy = accumulatedRewardGreedy / RoundsForAverage
            expectedRewardAdaptive = accumulatedRewardAdaptive / RoundsForAverage
            expectedRewardSingleK = accumulatedRewardSingleK / RoundsForAverage
            expectedRewardSafeRandomEta = accumulatedRewardSafeRandomEta / RoundsForAverage

            ## Average Runtime on a case
            expectedRuntimeOurs = accumulatedRuntimeOurs / RoundsForAverage
            expectedRuntimeGreedy = accumulatedRuntimeGreedy / RoundsForAverage
            expectedRuntimeAdaptive = accumulatedRuntimeAdaptive / RoundsForAverage
            expectedRuntimeSingleK = accumulatedRuntimeSingleK / RoundsForAverage
            expectedRuntimeSafeRandomEta = accumulatedRuntimeSafeRandomEta / RoundsForAverage

            print("accumulatedReward Ours: ", expectedRewardOurs)
            print("accumulatedReward Greedy: ", expectedRewardGreedy)
            print("accumulatedReward Single K: ", expectedRewardSingleK)
            print("accumulatedReward Adaptive: ", expectedRewardAdaptive)
            print("accumulatedReward SafeRandomEta: ", expectedRewardSafeRandomEta)

            AverageOverALengthOurs += expectedRewardOurs
            AverageOverALengthGreedy += expectedRewardGreedy
            AverageOverALengthSingleK += expectedRewardSingleK
            AverageOverALengthRandomEta += expectedRewardSafeRandomEta
            AverageOverALengthAdaptive += expectedRewardAdaptive

            AverageRuntimeALengthOurs += expectedRuntimeOurs
            AverageRuntimeALengthGreedy += expectedRuntimeGreedy
            AverageRuntimeALengthSingleK += expectedRuntimeSingleK
            AverageRuntimeALengthRandomEta += expectedRuntimeSafeRandomEta
            AverageRuntimeALengthAdaptive += expectedRuntimeAdaptive

        AverageOverALengthOurs /= caseNumberPerLength[traceLength-traceLengthRange[0]]
        AverageOverALengthGreedy /= caseNumberPerLength[traceLength-traceLengthRange[0]]
        AverageOverALengthSingleK /= caseNumberPerLength[traceLength-traceLengthRange[0]]
        AverageOverALengthRandomEta /= caseNumberPerLength[traceLength-traceLengthRange[0]]
        AverageOverALengthAdaptive /= caseNumberPerLength[traceLength-traceLengthRange[0]]

        AverageRuntimeALengthOurs /= caseNumberPerLength[traceLength - traceLengthRange[0]]
        AverageRuntimeALengthGreedy /= caseNumberPerLength[traceLength - traceLengthRange[0]]
        AverageRuntimeALengthSingleK /= caseNumberPerLength[traceLength - traceLengthRange[0]]
        AverageRuntimeALengthRandomEta /= caseNumberPerLength[traceLength - traceLengthRange[0]]
        AverageRuntimeALengthAdaptive /= caseNumberPerLength[traceLength - traceLengthRange[0]]

        AveragePerLengthListOurs.append(AverageOverALengthOurs)
        AveragePerLengthListGreedy.append(AverageOverALengthGreedy)
        AveragePerLengthListSingleK.append(AverageOverALengthSingleK)
        AveragePerLengthListRandomEta.append(AverageOverALengthRandomEta)
        AveragePerLengthListAdaptive.append(AverageOverALengthAdaptive)

        RuntimePerLengthListOurs.append(AverageRuntimeALengthOurs)
        RuntimePerLengthListGreedy.append(AverageRuntimeALengthGreedy)
        RuntimePerLengthListSingleK.append(AverageRuntimeALengthSingleK)
        RuntimePerLengthListRandomEta.append(AverageRuntimeALengthRandomEta)
        RuntimePerLengthListAdaptive.append(AverageRuntimeALengthAdaptive)

    # Average on all Trace
    for length in range(len(caseNumberPerLength)):

        TraceAverageOurs+=AveragePerLengthListOurs[length]*caseNumberPerLength[length]
        TraceAverageGreedy+=AveragePerLengthListGreedy[length]*caseNumberPerLength[length]
        TraceAverageSingleK+=AveragePerLengthListSingleK[length]*caseNumberPerLength[length]
        TraceAverageRandomEta+=AveragePerLengthListRandomEta[length]*caseNumberPerLength[length]
        TraceAverageAdaptive+=AveragePerLengthListAdaptive[length]*caseNumberPerLength[length]

        RuntimePerPromptOurs += RuntimePerLengthListOurs[length] * caseNumberPerLength[length]
        RuntimePerPromptGreedy += RuntimePerLengthListGreedy[length] * caseNumberPerLength[length]
        RuntimePerPromptSingleK += RuntimePerLengthListSingleK[length] * caseNumberPerLength[length]
        RuntimePerPromptRandomEta += RuntimePerLengthListRandomEta[length] * caseNumberPerLength[length]
        RuntimePerPromptAdaptive += RuntimePerLengthListAdaptive[length] * caseNumberPerLength[length]
    TraceAverageOurs/=sum(caseNumberPerLength)
    TraceAverageGreedy/=sum(caseNumberPerLength)
    TraceAverageSingleK/=sum(caseNumberPerLength)
    TraceAverageRandomEta/=sum(caseNumberPerLength)
    TraceAverageAdaptive/=sum(caseNumberPerLength)

    RuntimePerPromptOurs /= sum(caseNumberPerLength)
    RuntimePerPromptGreedy /= sum(caseNumberPerLength)
    RuntimePerPromptSingleK /= sum(caseNumberPerLength)
    RuntimePerPromptRandomEta /= sum(caseNumberPerLength)
    RuntimePerPromptAdaptive /= sum(caseNumberPerLength)

    print("AwardListPerLength Ours: ", AveragePerLengthListOurs)
    print("AwardListPerLength Greedy: ", AveragePerLengthListGreedy)
    print("AwardListPerLength Single K: ", AveragePerLengthListSingleK)
    print("AwardListPerLength Adaptive: ", AveragePerLengthListAdaptive)
    print("AwardListPerLength SafeRandomEta: ", AveragePerLengthListRandomEta)
    print("All Alg All Length:\n", [AveragePerLengthListOurs,AveragePerLengthListGreedy,AveragePerLengthListSingleK,AveragePerLengthListAdaptive,AveragePerLengthListRandomEta])

    print("AverageOverAllTraces Ours: ", TraceAverageOurs)
    print("AverageOverAllTraces Greedy: ", TraceAverageGreedy)
    print("AverageOverAllTraces Single K: ", TraceAverageSingleK)
    print("AverageOverAllTraces Adaptive: ", TraceAverageAdaptive)
    print("AverageOverAllTraces SafeRandomEta: ", TraceAverageRandomEta)

    print("\nAverageAllTraces with Poisson Every Alg:", [[TraceAverageOurs, TraceAverageGreedy, TraceAverageSingleK, TraceAverageAdaptive, TraceAverageRandomEta]])
    print("RuntimeAllTraces with Poisson Every Alg:", [[RuntimePerPromptOurs, RuntimePerPromptGreedy, RuntimePerPromptSingleK, RuntimePerPromptAdaptive, RuntimePerPromptRandomEta]])
