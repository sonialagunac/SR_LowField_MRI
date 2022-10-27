"""
Script used to analyze the data acquired from SynthSeg as a proxy for performance.
Sonia Laguna, 2022
"""
import os
import numpy as np
import csv
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from statistics import median


def read_csv(filename):
    with open(filename) as f:
        file_data=csv.reader(f)
        headers=next(file_data)
        return [dict(zip(headers,i)) for i in file_data]

# Loading and structuring results from SynthSeg
path_vol = './data_hyp/Exps'
exps = os.listdir(path_vol)

list = []
vols = {'vol_hip': [], #17 and 53
        'vol_put': [] ,#12 and 51
        'vol_amyg':[] ,#18 and 54
        'vol_pall':[],  #13 and 52
        'vol_caud':[],  #11 and 50
        'vol_thal':[], #10 and 49
        'vol_ventr':[], # 4, 5, 43, 44 right and left lateral and inferior
        'vol_cort':[], #3 and 42
        'vol_wmatter':[]} #2 and 41
labels = ['Hippocampus', 'Putamen', 'Amygdala', 'Pallidum', 'Caudate', 'Thalamus', 'Ventricles', 'Cerebral cortex', 'White matter']
pos = dict()
for j in range(len(exps)):
    csv = os.path.join(path_vol, exps[j])
    count = 0
    vols_hip = []  # 17 and 53
    vols_put = []  # 12 and 51
    vols_amyg = []  # 18 and 54
    vols_pall = []  # 13 and 52
    vols_caud = []  # 11 and 50
    vols_thal = []  # 10 and 49
    vols_ventr = []
    vols_cort = []
    vols_wmatter =[]

    for line in open(csv).readlines():
        count += 1
        if count is 2:
            for k in range(len(line.split(","))):
                if line.split(",")[k] == '17':
                    pos['vol_hip'] = [k]
                elif line.split(",")[k] == '53':
                    pos['vol_hip'].append(k)
                elif line.split(",")[k] == '12':
                    pos['vol_put'] = [k]
                elif line.split(",")[k] == '51':
                    pos['vol_put'].append(k)
                elif line.split(",")[k] == '18':
                    pos['vol_amyg'] = [k]
                elif line.split(",")[k] == '54':
                    pos['vol_amyg'].append(k)
                elif line.split(",")[k] == '13':
                    pos['vol_pall'] = [k]
                elif line.split(",")[k] == '52':
                    pos['vol_pall'].append(k)
                elif line.split(",")[k] == '11':
                    pos['vol_caud'] = [k]
                elif line.split(",")[k] == '50':
                    pos['vol_caud'].append(k)
                elif line.split(",")[k] == '10':
                    pos['vol_thal'] = [k]
                elif line.split(",")[k] == '49':
                    pos['vol_thal'].append(k)
                elif line.split(",")[k] == '4':
                    pos['vol_ventr'] = [k]
                elif line.split(",")[k] == '5':
                    pos['vol_ventr'].append(k)
                elif line.split(",")[k] == '43':
                    pos['vol_ventr'].append(k)
                elif line.split(",")[k] == '44':
                    pos['vol_ventr'].append(k)
                elif line.split(",")[k] == '3':
                    pos['vol_cort'] = [k]
                elif line.split(",")[k] == '42':
                    pos['vol_cort'].append(k)
                elif line.split(",")[k] == '2':
                    pos['vol_wmatter'] =[k]
                elif line.split(",")[k] == '41':
                    pos['vol_wmatter'].append(k)

        elif count >= 3:
            vols_hip.append(np.array(float(line.split(",")[pos['vol_hip'][0]])) + np.array(float(line.split(",")[pos['vol_hip'][1]])))
            vols_put.append(np.array(float(line.split(",")[pos['vol_put'][0]])) + np.array(float(line.split(",")[pos['vol_put'][1]])))
            vols_amyg.append(np.array(float(line.split(",")[pos['vol_amyg'][0]])) + np.array(float(line.split(",")[pos['vol_amyg'][1]])))
            vols_pall.append(np.array(float(line.split(",")[pos['vol_pall'][0]])) + np.array(float(line.split(",")[pos['vol_put'][1]])))
            vols_caud.append(np.array(float(line.split(",")[pos['vol_caud'][0]])) + np.array(float(line.split(",")[pos['vol_caud'][1]])))
            vols_thal.append(float(np.array(line.split(",")[pos['vol_thal'][0]])) + np.array(float(
                line.split(",")[pos['vol_thal'][1]])))
            vols_ventr.append(np.array(float(line.split(",")[pos['vol_ventr'][0]])) + np.array(float(
                line.split(",")[pos['vol_thal'][1]])))
            vols_cort.append(np.array(float(line.split(",")[pos['vol_cort'][0]])) + np.array(float(
                line.split(",")[pos['vol_thal'][1]])))
            vols_wmatter.append(np.array(float(line.split(",")[pos['vol_wmatter'][0]])) + np.array(float(
                line.split(",")[pos['vol_thal'][1]])))

    vols['vol_hip'].append(vols_hip)
    vols['vol_put'].append(vols_put)
    vols['vol_amyg'].append(vols_amyg)
    vols['vol_pall'].append(vols_pall)
    vols['vol_caud'].append(vols_caud)
    vols['vol_thal'].append(vols_thal)
    vols['vol_ventr'].append(vols_pall)
    vols['vol_cort'].append(vols_caud)
    vols['vol_wmatter'].append(vols_thal)


# Starting the analysis
corr = dict()
rmse = dict()
corr_tot=dict()
rmse_tot=dict()
plot = False
plt.rcParams['font.size'] = '8'
for w in range(len(exps)-1):
    count = 0
    corr_tot[exps[w + 1][0:2]] = dict()
    rmse_tot[exps[w + 1][0:2]] = dict()
    if plot:
        fig, ax = plt.subplots(3,3, constrained_layout = True)
    for name in vols:
        vol_var = name
        if count == 0:
            nums = [0,0]
        elif count == 1:
            nums = [0, 1]
        elif count == 2:
            nums = [1, 1]
        elif count == 3:
            nums = [0, 2]
        elif count == 4:
            nums = [1, 0]
        elif count == 5:
            nums = [1, 2]
        elif count == 6:
            nums = [2, 2]
        elif count == 7:
            nums = [2, 0]
        elif count == 8:
            nums = [2, 1]

        # Computing pearson correlation and RMSE of volumes
        CORR, _ = pearsonr(vols[vol_var][0], vols[vol_var][w + 1])
        corr[vol_var] = CORR
        RMSE = np.sqrt(np.mean(np.power(np.asarray(vols[vol_var][0]) - np.asarray(vols[vol_var][w + 1]), 2)))
        rmse[vol_var] = RMSE
        corr_tot[exps[w+1][0:2]][vol_var] = CORR
        rmse_tot[exps[w + 1][0:2]][vol_var] = RMSE
        if plot:
            theta = np.polyfit(vols[vol_var][0], vols[vol_var][w + 1], 1)
            y_line = theta[1] + theta[0] * np.asarray(vols[vol_var][0])

            ax[nums[0], nums[1]].scatter(vols[vol_var][0], vols[vol_var][w + 1], s = 5)
            ax[nums[0], nums[1]].plot(vols[vol_var][0], y_line, 'b', linewidth= 0.5)
            ax[nums[0], nums[1]].set_title(labels[count] + ' r: ' + str(np.around(corr[vol_var], decimals=2)) + ' RMSE: ' + str(
                np.around(rmse[vol_var], decimals=2)), fontsize = 8)
            ax[nums[0], nums[1]].set_xlabel(exps[0][:-4],  fontsize = 8)
            ax[nums[0], nums[1]].set_ylabel(exps[w + 1][:-4],  fontsize = 8)
            lims = [
                np.min([ax[nums[0], nums[1]].get_xlim(),ax[nums[0], nums[1]].get_ylim()]),  # min of both axes
                np.max([ax[nums[0], nums[1]].get_xlim(), ax[nums[0], nums[1]].get_ylim()]),  # max of both axes
            ]

            ax[nums[0], nums[1]].plot(lims, lims, 'k--', alpha=0.75, zorder=0, linewidth= 0.5)
            ax[nums[0], nums[1]].set_aspect('equal')
            ax[nums[0], nums[1]].set_xlim(lims)
            ax[nums[0], nums[1]].set_ylim(lims)
            count += 1
    if plot:
        plt.show()

"""
Focusing on the ablation experiments
"""

# Structuring the experiments IDs
configs =read_csv('./data_hyp/exps_short.csv') # File with IDs of all experiments
plot = True
count_e=0
corrs=[]
rmses=[]
titles=[]
for k in range(len(exps)-3):
    for i in range(len(configs)):
        if configs[i]['var'] =='T1':
            print('Experiment with T1:', int(configs[i][exps[k + 3][0:2]]) )
            if not int(configs[i][exps[k + 3][0:2]]):
                t1 = 1
                break
        elif configs[i]['var'] =='dataugm':
            print('Experiment with data augm:', int(configs[i][exps[k + 3][0:2]]))
            if not int(configs[i][exps[k + 3][0:2]]):
                dataugm = 1
                break
        elif configs[i]['var'] == 'cycle':
            print('Experiment with cycle:', int(configs[i][exps[k + 3][0:2]]))
            if int(configs[i][exps[k + 3][0:2]]):
                cycle = 0
                break
        elif configs[i]['var'] == 'denois':
            print('Experiment with denois:', int(configs[i][exps[k + 3][0:2]]))
            if not int(configs[i][exps[k + 3][0:2]]):
                denois =1
                break
    else:
        print('Experiment number: ', exps[k + 3][0:2])
        corrs.append([])
        rmses.append([])
        if plot:
            count = 0
            fig, ax = plt.subplots(3, 3, constrained_layout=True)
        for name in vols:
            vol_var = name
            corrs[count_e].append(corr_tot[exps[k + 3][0:2]][vol_var])
            rmses[count_e].append(rmse_tot[exps[k + 3][0:2]][vol_var])
            if plot:
                if count == 0:
                    nums = [0, 0]
                elif count == 1:
                    nums = [0, 1]
                elif count == 2:
                    nums = [1, 1]
                elif count == 3:
                    nums = [0, 2]
                elif count == 4:
                    nums = [1, 0]
                elif count == 5:
                    nums = [1, 2]
                elif count == 6:
                    nums = [2, 2]
                elif count == 7:
                    nums = [2, 0]
                elif count == 8:
                    nums = [2, 1]
                theta = np.polyfit(vols[vol_var][0], vols[vol_var][k + 3], 1)
                y_line = theta[1] + theta[0] * np.asarray(vols[vol_var][0])

                ax[nums[0], nums[1]].scatter(vols[vol_var][0], vols[vol_var][k + 3], s=5)
                ax[nums[0], nums[1]].plot(vols[vol_var][0], y_line, 'b', linewidth=0.5)
                ax[nums[0], nums[1]].set_title(labels[count] + ' r: ' + str(
                    np.around(corr_tot[exps[k + 3][0:2]][vol_var], decimals=2)) + ' RMSE: ' + str(
                    np.around(rmse_tot[exps[k + 3][0:2]][vol_var], decimals=2)), fontsize=8)
                ax[nums[0], nums[1]].set_xlabel(exps[0][:-4], fontsize=8)
                ax[nums[0], nums[1]].set_ylabel(exps[w + 1][:-4], fontsize=8)
                lims = [
                    np.min([ax[nums[0], nums[1]].get_xlim(), ax[nums[0], nums[1]].get_ylim()]),  # min of both axes
                    np.max([ax[nums[0], nums[1]].get_xlim(), ax[nums[0], nums[1]].get_ylim()]),  # max of both axes
                ]

                ax[nums[0], nums[1]].plot(lims, lims, 'k--', alpha=0.75, zorder=0, linewidth=0.5)
                ax[nums[0], nums[1]].set_aspect('equal')
                ax[nums[0], nums[1]].set_xlim(lims)
                ax[nums[0], nums[1]].set_ylim(lims)
                count += 1
        if plot:
            titles.append(exps[k + 3][0:2])
            fig.suptitle('Experiment number: ' + exps[k + 3][0:2])
            plt.show()
        count_e+=1


"""
Choosing best experiment based on median correlation:
"""

titles_2 = ['w1e-3\nlr1e-4','w1e-4\nlr1e-4','w1e-2\nlr1e-4', 'w1e-3\nlr1e-5' ]
fig1, ax1 = plt.subplots(1,2)
ax1[0].set_title('Correlation coefficient')
ax1[0].boxplot(corrs) #Use list(corrs.values)
ax1[0].set_xticklabels(titles_2)
ax1[0].set_ylim([0,1])
for i in range(len(corrs)):
    ax1[0].text(i + 1, 0.05, np.around(median(corrs[i]), decimals=2),
             ha='center')
ax1[1].tick_params()
ax1[1].set_title('RMSE')
ax1[1].boxplot(corrs.values) # Can also focus on the RMSE
ax1[1].set_xticklabels(titles_2)
for i in range(len(corrs)):
    ax1[1].text(i + 1, np.amin(rmses)-80, np.around(median(corrs.values[i]), decimals=2),
             ha='center')
ax1[1].tick_params()
if t1:
    t='T1, '
else:
    t='T2, '
if denois:
    d='with denoiser, '
else:
    d='without denoiser, '
if dataugm:
    aug='with segm data augm, '
else:
    aug = 'without segm data augm, '
if cycle:
    cy='CycleGAN '
else:
    cy='FastCUT '
fig1.suptitle(t + d+ aug + cy)
plt.show()
