import numpy as np
import matplotlib.pyplot as plt
width = 0.5

colum1 = [0.8244695657173011, 0.8358239410416131, 0.8518668500407482, 0.8436267502420695, 0.843476195730415, 0.8295709255034865, 0.8168742506052928,
          0.8940786785051442, 0.8347562864016531, 0.8942864809662239, 0.8365695236159006, 0.8852949790528576, 0.8654831650929691, 0.8557260809150765, 0.8590348736166005]
colum2 = [0.8172457267294957, 0.8280442509581938, 0.7966266984829235, 0.8928875988702782, 0.8277732648413471, 0.9257837267558477, 0.793877572663102,
          0.9279074910577957, 0.793271386871652, 0.8197129223666477, 0.7992054708556672, 0.851268216562193, 0.8246731414336695, 0.910244761128153, 0.8748561815038054]
colum3 = [0.9506539617188056, 0.8182973234523616, 0.881407193860745, 0.7428458817931004, 0.7653685356531317, 0.7432880131689693, 0.9427797625123328,
          0.7667097591018459, 0.808172783508261, 0.868444176166753, 0.7812860798892429, 0.8084694732013797, 0.931413659930098, 0.9763064889166083, 0.7597957219656332]
colum4 = [0.8551665035896748, 0.944353706742627, 0.9353133132829028, 0.8322205029013526, 0.8902763636458206, 0.8218586035686991, 0.9258505852677578,
          0.8203641738896712, 0.828451155669825, 0.802762891320745, 0.7610548999380791, 0.8187008717741007, 0.8583581579593613, 0.8255357820893472, 0.8276985146175407]
colum5 = [0.7300449625398395, 0.8405173468270977, 0.8593569557865204, 0.8804846358734486, 0.8277175213592279, 0.8493690940770477, 0.7636975013290793,
          0.8344151534186464, 0.9142715856386865, 0.7857255762611118, 0.7737075540953362, 0.9376000771930917, 0.8776414159494049, 0.8627640722378183, 0.8299735891409866]



colum = [colum1,colum2,colum3,colum4, colum5]
btm = [0.8155620957507762, 0.784736745978102, 0.755076484913643, 0.7265372763434386, 0.6990767484659679]

labels = ['3', '4', '5', '6', '7']
x = np.arange(len(labels))

fig, ax = plt.subplots()
size = 12
ax.tick_params(axis='both', which='major', labelsize=size)
ax.tick_params(axis='both', which='minor', labelsize=size)
plt.rc('font', size=size)  # controls default text sizes
plt.rc('axes', titlesize=size)  # fontsize of the axes title
plt.rc('axes', labelsize=size)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
plt.rc('ytick', labelsize=size)  # fontsize of the tick labels
plt.rc('legend', fontsize=14)  # legend fontsize
plt.rc('figure', titlesize=size)
for i in range(15):
    scy = []
    for j in x:
        scy.append(colum[j][i])
    if i == 0:
        ax.scatter(x, scy, color='#E50000', edgecolor='k', marker='o', zorder=1, label="Experimental")
    else:
        ax.scatter(x, scy, color='#E50000', edgecolor='k', marker='o', zorder=1)
ax.bar(x, 1, width, edgecolor='k', zorder=0, color = '#15B01A')
ax.bar(x, btm, width, label="Theoretical", edgecolor='k', color = '#929591')
ax.set_ylabel(r"$\mathbb{E}\left[ALG\right]$ / $\max$ OPT", fontsize=18)
ax.set_xlabel(r"$M$", fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1))
ax.set_ylim(0.6, 1.005)
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('pdf',fonttype = 42)
plt.rc('ps',fonttype = 42)
plt.style.use('classic')
plt.tight_layout()
figure = plt.gcf()
figure.savefig("Figure1_c.pdf")
plt.show()