import matplotlib.pyplot as plt

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
    plt.ylabel(r"Average Accumulated Reward (Satisfaction)", fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.xticks([(i+1)*offset for i in range(bars)], labels,fontsize=10)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)

    figure = plt.gcf()
    figure.savefig(filename)
    plt.show()

def drawDistOriginal():
    RawData = [[1.778838276580275, 1.280515863151447, 1.5245885309630318, 1.280515863151447, 1.280515863151447],
               [1.587687252482042, 1.1127906762125404, 1.2629895021543391, 1.1127906762125404, 1.2804976824370315],
               [1.778838276580275, 1.280515863151447, 1.5245885309630318, 1.280515863151447, 1.280515863151447],
               [1.587687252482042, 1.1127906762125404, 1.2629895021543391, 1.1127906762125404, 1.206030686580663]]
    fileName = "Original Dist 4 ST.pdf"
    xLabel = "Average Performance Original Distribution"
    return RawData,fileName, xLabel

def drawDistEven():
    RawData = [[1.7667977233350427, 1.218047711182537, 1.44174458486531, 1.218047711182537, 1.218047711182537],
               [1.5419903267897785, 1.0748696515255844, 1.211844427263494, 1.0748696515255844, 1.237307776810429],
               [1.7667977233350427, 1.218047711182537, 1.44174458486531, 1.218047711182537, 1.218047711182537],
               [1.5419903267897785, 1.0748696515255844, 1.211844427263494, 1.0748696515255844, 1.151103631586067]]
    fileName = "Uniform Dist 4 ST.pdf"
    xLabel = "Average Performance Uniform Distribution"
    return RawData,fileName, xLabel

def drawDistLogNorm():
    RawData = [[2.265324404795198, 0.35419951157145657, 1.0074498085813777, 0.35419951157145657, 0.35419951157145657],
               [1.9522186833520039, 0.16066412192754403, 0.6423060461829604, 0.16066412192754403, 0.562083597211641],
               [2.265324404795198, 0.35419951157145657, 1.0074498085813777, 0.35419951157145657, 0.35419951157145657],
               [1.9522186833520039, 0.16066412192754403, 0.6423060461829604, 0.16066412192754403, 0.4970176717668425]]
    fileName = "Log Norm Dist 4 ST.pdf"
    xLabel = "Average Performance Log Norm Distribution"
    return RawData,fileName, xLabel


def drawDistPoisson():
  RawData = [[2.08968627777958, 0.43970072670197363, 1.1244329844206045, 0.43970072670197363, 0.43970072670197363],
             [1.8294346749591022, 0.17894533615962757, 0.7603005480064876, 0.17894533615962757, 0.5559880320005192],
             [2.08968627777958, 0.43970072670197363, 1.1244329844206045, 0.43970072670197363, 0.43970072670197363],
             [1.8294346749591022, 0.17894533615962757, 0.7603005480064876, 0.17894533615962757, 0.6066765903319443]]
  fileName = "Poisson Dist 4 ST.pdf"
  xLabel = "Average Performance Poisson Distribution"
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