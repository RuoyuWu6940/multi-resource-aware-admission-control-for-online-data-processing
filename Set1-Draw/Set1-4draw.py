import numpy as np
import matplotlib.pyplot as plt
width = 0.5

colum1 = [0.7829557264047364, 0.7719483087695975, 0.8100967677481409, 0.7910225382588691, 0.7833153757554479, 0.7934432760964333, 0.7697448550837456,
          0.7746752863964149, 0.7731727392520749, 0.8414884852289221, 0.7803212197626231, 0.8582470505100153, 0.7738303204738486, 0.8111125332650706, 0.7790600998799009]
colum2 = [0.7308955255731072, 0.716955685094261, 0.762785139408824, 0.7520672295556344, 0.7318199180476593, 0.7151714290540052, 0.7653301380009452,
          0.7631687250712519, 0.745743234494367, 0.7914987924494935, 0.7670729303463868, 0.7324816334922845, 0.7926163651638481, 0.7239222789181886, 0.7747639012077465]
colum3 = [0.6723088235672209, 0.7224890310702077, 0.7406780151699379, 0.7141829218833814, 0.6569933546233416, 0.7169097378840271, 0.6637108901931035,
          0.6768747101098058, 0.6693346961774833, 0.6658056725860007, 0.678535794462459, 0.6574295801524243, 0.7318429677292202, 0.6952029538934013, 0.7084664806671954]
colum4 = [0.6667981082619328, 0.6741232897729924, 0.6815157577104676, 0.6501583030994796, 0.721999674582908, 0.7206619111882265, 0.6579791279229532,
          0.7146584464908161, 0.66508047792498, 0.668688821527927, 0.7236051347803768, 0.6584393226754243, 0.6805368094759311, 0.6087572262543796, 0.7381863485546816]
colum5 = [0.6372903463064294, 0.7115177797282898, 0.6355695059104881, 0.6675029873168704, 0.63523276475486, 0.6853399725057167, 0.6616729236802573,
          0.7023343426245203, 0.5922433528425876, 0.6667467387315245, 0.688798127858517, 0.6689422513950235, 0.6486854857081078, 0.736551410668485, 0.679365312910182]



colum = [colum1,colum2,colum3,colum4, colum5]
btm = [0.7666932351329182, 0.7037306036526222, 0.6459386098945128, 0.5928926290640806, 0.5442029075424425]

labels = ['1', '2', '3', '4', '5']
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
ax.set_xlabel(r"$N$", fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1))
ax.set_ylim(0.5, 1.005)
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('pdf',fonttype = 42)
plt.rc('ps',fonttype = 42)
plt.style.use('classic')
plt.tight_layout()
figure = plt.gcf()
figure.savefig("Figure1_d.pdf")
plt.show()