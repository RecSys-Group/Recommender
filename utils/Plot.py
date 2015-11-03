import numpy as np
import pylab as pl
x = [50, 100, 150, 200]
y_top_f1 = [0.0439, 0.0439, 0.0439, 0.0439]
y_nmf_f1 = [0.0508, 0.0571, 0.0610, 0.0634]
y_bpr_f1 = [0.0520, 0.0586, 0.0618, 0.0621]
y_bprnn_f1 = [0.0654, 0.0677, 0.0701, 0.0716]
y_bprseq_f1 = [0.0545, 0.0592, 0.0653, 0.0663]
y_hlbpr_f1 = [0.0656, 0.0697, 0.0714, 0.0736]

y_top_hr = [0.3451, 0.3451, 0.3451, 0.3451]
y_nmf_hr = [0.3810, 0.4151, 0.4356, 0.4464]
y_bpr_hr = [0.3890, 0.4261, 0.4380, 0.4412]
y_bprnn_hr = [0.4531, 0.4610, 0.4710, 0.4827]
y_bprseq_hr = [0.400, 0.426, 0.455, 0.459]
y_hlbpr_hr = [0.455, 0.4730, 0.4836, 0.4957]

y_top_ndgg = [0.1133, 0.1133, 0.1133, 0.1133]
y_nmf_ndgg = [0.1243, 0.1365, 0.1435, 0.1480]
y_bpr_ndgg = [0.1191, 0.1300, 0.1356, 0.1345]
y_bprnn_ndgg = [0.144, 0.150, 0.157, 0.1590]
y_bprseq_ndgg = [0.125, 0.137, 0.151, 0.1525]
y_hlbpr_ndgg = [0.147, 0.1540, 0.1598, 0.1632]

bbg_y_top_f1 = [0.0309, 0.0309, 0.0309, 0.0309]
bbg_y_nmf_f1 = [0.0339, 0.0367, 0.0382, 0.0391]
bbg_y_bpr_f1 = [0.0355, 0.0386, 0.0409, 0.0412]
bbg_y_bprnn_f1 = [0.0389, 0.0420, 0.0423, 0.0424]
bbg_y_bprseq_f1 = [0.0373, 0.0419, 0.0420, 0.0421]
bbg_y_hlbpr_f1 = [0.0404, 0.0430, 0.0432, 0.0438]

bbg_y_top_hr = [0.181, 0.181, 0.181, 0.181]
bbg_y_nmf_hr = [0.183, 0.1992, 0.2000, 0.2051]
bbg_y_bpr_hr = [0.19, 0.2028, 0.2109, 0.2106]
bbg_y_bprnn_hr = [0.209, 0.2158, 0.2159, 0.2210]
bbg_y_bprseq_hr = [0.19, 0.2101, 0.2105, 0.2110]
bbg_y_hlbpr_hr = [0.219, 0.2296, 0.2300, 0.2440]

bbg_y_top_ndgg = [0.0386, 0.0386, 0.0386, 0.0386]
bbg_y_nmf_ndgg = [0.0487, 0.0533, 0.0544, 0.0552]
bbg_y_bpr_ndgg = [0.050, 0.0536, 0.0555, 0.0559]
bbg_y_bprnn_ndgg = [0.0528, 0.0566, 0.0570, 0.0571]
bbg_y_bprseq_ndgg = [0.0512, 0.0560, 0.0569, 0.0565]
bbg_y_hlbpr_ndgg = [0.0540, 0.0587, 0.0592, 0.0593]

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 25,
        }
tafeng1 = pl.subplot(231)
plot01, = tafeng1.plot(x, y_top_f1, 'y+-', linewidth=3)# use pylab to plot x and y : Give your plots names
plot02, = tafeng1.plot(x, y_nmf_f1, 'gp-', linewidth=3)
plot03, = tafeng1.plot(x, y_bpr_f1, 'r*-', linewidth=3)
plot04, = tafeng1.plot(x, y_bprnn_f1, 'ch:', linewidth=3)
plot05, = tafeng1.plot(x, y_bprseq_f1, 'mH-.', linewidth=3)
plot06, = tafeng1.plot(x, y_hlbpr_f1, 'bs--', linewidth=3)
pl.xticks(fontsize=17)
pl.yticks(fontsize=17)
#pl.title('Ta-Feng', fontdict=font)# give plot a title
pl.xlabel('dimensionality', fontdict=font)# make axis labels
pl.ylabel('F1-score@Top5', fontdict=font)
pl.xlim(40.0, 210.0)# set axis limits
pl.xticks([50.0, 100.0, 150.0, 200.0])
pl.ylim(0.04, 0.075)
pl.yticks([.04, .05, .06, .07, .08])
#pl.legend([plot1, plot2, plot3, plot4, plot5, plot6], ('TOP', 'NMF', 'BPR', 'NSEQ_BPR', 'SEQ_BPR', 'HLBPR'), 'best', numpoints=1)# make legend

tafeng3 = pl.subplot(233)
plot21, = tafeng3.plot(x, y_top_ndgg, 'y+-', linewidth=3)# use pylab to plot x and y : Give your plots names
plot22, = tafeng3.plot(x, y_nmf_ndgg, 'gp-', linewidth=3)
plot23, = tafeng3.plot(x, y_bpr_ndgg, 'r*-', linewidth=3)
plot24, = tafeng3.plot(x, y_bprnn_ndgg, 'ch:', linewidth=3)
plot25, = tafeng3.plot(x, y_bprseq_ndgg, 'mH-.', linewidth=3)
plot26, = tafeng3.plot(x, y_hlbpr_ndgg, 'bs--', linewidth=3)
pl.xticks(fontsize=17)
pl.yticks(fontsize=17)
#pl.title('Ta-Feng', fontdict=font)
pl.xlabel('dimensionality', fontdict=font)# make axis labels
pl.ylabel('NDGG@Top5', fontdict=font)
pl.xlim(40.0, 210.0)# set axis limits
pl.xticks([50.0, 100.0, 150.0, 200.0])
pl.ylim(0.1, 0.17)
pl.yticks([0.1, 0.12, 0.14, 0.16])
bbg1 = pl.subplot(234)
plot31, = bbg1.plot(x, bbg_y_top_f1, 'y+-', linewidth=3)# use pylab to plot x and y : Give your plots names
plot32, = bbg1.plot(x, bbg_y_nmf_f1, 'gp-', linewidth=3)
plot33, = bbg1.plot(x, bbg_y_bpr_f1, 'r*-', linewidth=3)
plot34, = bbg1.plot(x, bbg_y_bprnn_f1, 'ch:', linewidth=3)
plot35, = bbg1.plot(x, bbg_y_bprseq_f1, 'mH-.', linewidth=3)
plot36, = bbg1.plot(x, bbg_y_hlbpr_f1, 'bs--', linewidth=3)
pl.xticks(fontsize=17)
pl.yticks(fontsize=17)
#pl.title('BBG', fontdict=font)# give plot a title
pl.xlabel('dimensionality', fontdict=font)# make axis labels
pl.ylabel('F1-score@Top5', fontdict=font)
pl.xlim(40.0, 210.0)# set axis limits
pl.xticks([50.0, 100.0, 150.0, 200.0])
pl.ylim(0.030, 0.045)
pl.yticks([0.030, 0.035, 0.04, 0.045])


bbg2 = pl.subplot(235)
plot41, = bbg2.plot(x, bbg_y_top_hr, 'y+-', linewidth=3)# use pylab to plot x and y : Give your plots names
plot42, = bbg2.plot(x, bbg_y_nmf_hr, 'gp-', linewidth=3)
plot43, = bbg2.plot(x, bbg_y_bpr_hr, 'r*-', linewidth=3)
plot44, = bbg2.plot(x, bbg_y_bprnn_hr, 'ch:', linewidth=3)
plot45, = bbg2.plot(x, bbg_y_bprseq_hr, 'mH-.', linewidth=3)
plot46, = bbg2.plot(x, bbg_y_hlbpr_hr, 'bs--', linewidth=3)
pl.xticks(fontsize=17)
pl.yticks(fontsize=17)
pl.title('BBG', fontdict=font)
pl.ylabel('Hit-Ratio@Top5', fontdict=font)# give plot a title
pl.xlabel('dimensionality', fontdict=font)# make axis labels
pl.xlim(40.0, 210.0)# set axis limits
pl.xticks([50.0, 100.0, 150.0, 200.0])
pl.ylim(0.15, 0.25)
pl.yticks([0.15, 0.18, 0.21, 0.24])

bbg3 = pl.subplot(236)
plot51, = bbg3.plot(x, bbg_y_top_ndgg, 'y+-', linewidth=3)# use pylab to plot x and y : Give your plots names
plot52, = bbg3.plot(x, bbg_y_nmf_ndgg, 'gp-', linewidth=3)
plot53, = bbg3.plot(x, bbg_y_bpr_ndgg, 'r*-', linewidth=3)
plot54, = bbg3.plot(x, bbg_y_bprnn_ndgg, 'ch:', linewidth=3)
plot55, = bbg3.plot(x, bbg_y_bprseq_ndgg, 'mH-.', linewidth=3)
plot56, = bbg3.plot(x, bbg_y_hlbpr_ndgg, 'bs--', linewidth=3)
pl.xticks(fontsize=17)
pl.yticks(fontsize=17)
#pl.title('BBG', fontdict=font)
pl.ylabel('NDGG@Top5', fontdict=font)# give plot a title
pl.xlabel('dimensionality', fontdict=font)# make axis labels
pl.xlim(40.0, 210.0)# set axis limits
pl.xticks([50.0, 100.0, 150.0, 200.0])
pl.ylim(0.03, 0.065)
pl.yticks([0.03, 0.04, 0.05, 0.06])
tafeng2 = pl.subplot(232)
plot11, = tafeng2.plot(x, y_top_hr, 'y+-', linewidth=3)# use pylab to plot x and y : Give your plots names
plot12, = tafeng2.plot(x, y_nmf_hr, 'gp-', linewidth=3)
plot13, = tafeng2.plot(x, y_bpr_hr, 'r*-', linewidth=3)
plot14, = tafeng2.plot(x, y_bprnn_hr, 'ch:', linewidth=3)
plot15, = tafeng2.plot(x, y_bprseq_hr, 'mH-.', linewidth=3)
plot16, = tafeng2.plot(x, y_hlbpr_hr, 'bs--', linewidth=3)
pl.xticks(fontsize=17)
pl.yticks(fontsize=17)
pl.title('Ta-Feng', fontdict=font)
pl.ylabel('Hit-Ratio@Top5', fontdict=font)# give plot a title
pl.xlabel('dimensionality', fontdict=font)# make axis labels
pl.legend([plot11, plot12, plot13, plot14, plot15, plot16], ('TOP', 'NMF', 'BPR', 'NSP_BPR', 'SP_BPR', 'HLBPR'), bbox_to_anchor=(0., 1.19, 1., .102), fontsize=15, borderaxespad=0., ncol=6, loc=9)
pl.xlim(40.0, 210.0)# set axis limits
pl.xticks([50.0, 100.0, 150.0, 200.0])
pl.ylim(0.3, 0.53)

pl.subplots_adjust(left=0.08, right=0.95, wspace=0.4, hspace=0.45)
pl.show()# show the plot on the screen
