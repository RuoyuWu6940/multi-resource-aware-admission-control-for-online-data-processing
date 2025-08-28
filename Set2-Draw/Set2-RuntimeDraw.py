import matplotlib.pyplot as plt
import numpy as np

def testDraw(
        data,labels,xLabel,filename,bars,benchmarks,legend_labels
):

    bar_width = 0.43
    offset = 6

    positions = []
    for i in range(bars):
        for j in range(benchmarks):
            positions.append((i+1)*offset-bars*bar_width+j*2*bar_width)

    color = ['#008000', '#F0E68C', '#808080','#000080' ,'#ADD8E6']
    plt.bar(positions, data, color=color, edgecolor='black', linewidth=2)

    legend_labels = legend_labels
    legend_handles = [plt.Line2D([0], [0], color=color, linewidth=3, linestyle='-') for color in
                      color]
    plt.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.15),
               fancybox=True, shadow=True, ncol=6, fontsize=10)

    plt.xlabel(xLabel, fontsize=12)
    plt.ylabel(r"Average Runtime for Each Task (second)", fontsize=12)
    plt.yscale('log')

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.xticks([(i+1)*offset for i in range(bars)], labels,fontsize=10)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)

    figure = plt.gcf()
    figure.savefig(filename)
    plt.show()

def drawDistOriginal():
    RawData = [[3.155304591795317e-05, 7.396190553564923e-07, 1.8811410095368722e-05, 7.855325696929727e-07, 7.852021683551508e-07],
               [6.34133074070841e-05, 1.590260668281135e-06, 2.0136814782524116e-05, 1.606319962550433e-06, 1.644012654004438e-06],
               [5.34334740630026e-05, 8.562709244141602e-07, 2.1487142251766416e-05, 9.075137891320168e-07, 9.112546676117804e-07],
               [8.478075958947892e-05, 1.7137272111224728e-06, 2.0592739433598626e-05, 1.7182078977325593e-06, 1.7916220945173074e-06]]
    fileName = "Original Dist 4 ST Runtime.pdf"
    xLabel = "Average Runtime Original Distribution"
    return RawData,fileName, xLabel

def drawDistEven():
    RawData = [[np.float64(3.241508149114704e-05), np.float64(7.135501934750007e-07), np.float64(1.9265917908497813e-05), np.float64(7.589701781399666e-07), np.float64(7.596283816683784e-07)],
               [np.float64(6.466177784256049e-05), np.float64(1.5085871916059912e-06), np.float64(2.0469352704269796e-05), np.float64(1.5338715095069364e-06), np.float64(1.5682673746234731e-06)],
               [np.float64(5.4321816773058236e-05), np.float64(8.173454154848684e-07), np.float64(2.1667644959380015e-05), np.float64(8.674895309319448e-07), np.float64(8.701806920741014e-07)],
               [np.float64(8.679121677861725e-05), np.float64(1.6322409232998979e-06), np.float64(2.095439976465629e-05), np.float64(1.643579629756621e-06), np.float64(1.7112908770687997e-06)]]
    fileName = "Uniform Dist 4 ST Runtime.pdf"
    xLabel = "Average Runtime Uniform Distribution"
    return RawData,fileName, xLabel

def drawDistLogNorm():
    RawData = [[3.23524021322646e-05, 7.06114096680376e-07, 1.9209109815783883e-05, 7.710846674134808e-07, 7.70438432686128e-07],
               [7.751208462222494e-05, 1.6905445431655035e-06, 2.6349957386395152e-05, 1.679863629189575e-06, 1.859026333228928e-06],
               [5.0033141543509295e-05, 7.818366541858479e-07, 1.975711084128346e-05, 8.420204901054434e-07, 8.453810829321718e-07],
               [9.266203417559362e-05, 1.7592905052892876e-06, 2.28575004435315e-05, 1.795458677502612e-06, 1.847905341687011e-06]]
    fileName = "Log Norm Dist 4 ST Runtime.pdf"
    xLabel = "Average Runtime Log Norm Distribution"
    return RawData,fileName, xLabel


def drawDistPoisson():
  RawData = [[3.4223394310512594e-05, 7.568308965936419e-07, 2.053331162744142e-05, 8.059840727340558e-07, 8.223773329962173e-07],
             [6.75926197113156e-05, 1.6557254930079565e-06, 2.1526938026673108e-05, 1.6589553151881746e-06, 1.7316260444564083e-06],
             [5.344603231227201e-05, 8.66282279208816e-07, 2.0788873166480493e-05, 9.060983640301939e-07, 8.984185615209801e-07],
             [0.00010044348668975713, 1.8840961712953795e-06, 2.5081530828479875e-05, 1.809087315409794e-06, 1.9520823225391665e-06]]
  fileName = "Poisson Dist 4 ST Runtime.pdf"
  xLabel = "Average Runtime Poisson Distribution"
  return RawData, fileName, xLabel

# def drawDistOriginal():

if __name__ == '__main__':
    # RawData, fileName, xLabel = drawDistOriginal()
    # RawData, fileName, xLabel = drawDistEven()
    RawData, fileName, xLabel = drawDistLogNorm()
    # RawData, fileName, xLabel = drawDistPoisson()

    Traces = 4
    numAlg = 5
    rows = [[] for i in range(numAlg)]
    for i in range(numAlg):
      for j in range(Traces):
        rows[i].append(RawData[j][i])
    print(len(rows[0]))
    data = []
    for j in range(Traces):
        for i in rows:
            data.append(i[j])
    print(data)
    labels = ['Situation 1','Situation 2','Situation 3',"Situation 4"]

    legend_label = ['OMMA','G','OSMA',"Ada","R"]
    testDraw(data,labels,xLabel,fileName,Traces,numAlg,legend_label)