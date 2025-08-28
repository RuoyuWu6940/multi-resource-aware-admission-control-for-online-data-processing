import numpy as np
import matplotlib.pyplot as plt
width = 0.5

colum1 = [0.8252380152439132, 0.8273515985270166, 0.8271087764693037, 0.8184915941396612, 0.8144801975261364, 0.8239084394822598, 0.8192730479082969,
          0.824236586897097, 0.8308367523407042, 0.8303158738273919, 0.816837706096588, 0.8336321624606621, 0.8200243027173335, 0.8152029751294453, 0.8315248354841913]
colum2 = [0.8874358225474215, 0.8948404005321299, 0.88114978276498, 0.8784793499151538, 0.8953667066466364, 0.8855957241441992, 0.893320697710557,
          0.8895865977532987, 0.8944510390331126, 0.8993918748819936, 0.8849218459961173, 0.8849242825176654, 0.8941939052295009, 0.8958623503425763, 0.8957226028599158]
colum3 = [0.9260362913012731, 0.9201235381430677, 0.9143982254374566, 0.9199539852179428, 0.9138271489157718, 0.9190170368388181, 0.9208869458396952,
          0.9115856056939344, 0.914803520960984, 0.9189167579083656, 0.9221766571152643, 0.9133623052227694, 0.9164751048809019, 0.9148052625923024, 0.9153466128687483]
colum4 = [0.9304838925095184, 0.9362616686210976, 0.9400355438600723, 0.9461097873656031, 0.9351276041666663, 0.9466402487381884, 0.9286991604815574,
          0.9416005109955281, 0.9373052042854741, 0.9370131344592179, 0.9440492185095182, 0.9417456382953284, 0.9333093086505831, 0.935379221119645, 0.9271616193343589]


colum = [colum1,colum2,colum3,colum4] # re-arrange the order of data
btm = [0.7089206845641766, 0.7496270472484637, 0.7747120027954064, 0.7921886060757515]  # the theoretical competitive ratios

labels = ['200', '300', '400', '500']
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
ax.set_xlabel(r"$\ K_m,\forall m$", fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1))
ax.set_ylim(0.7, 1.005)

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('pdf',fonttype = 42)
plt.rc('ps',fonttype = 42)

plt.style.use('classic')
plt.tight_layout()
figure = plt.gcf()
figure.savefig("Figure1_a.pdf")
plt.show()