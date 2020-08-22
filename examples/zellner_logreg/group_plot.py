import bokeh.plotting as bkp
from bokeh.io import export_png
import pickle as pk
import numpy as np
import sys, os
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *
import bokeh.layouts as bkl
from bokeh.models import Title

dnm='diabetes'
if dnm=='adult':
  dnmnm='ADULT'
if dnm=='diabetes':
  dnmnm='HOSPITAL READMISSIONS'
fldr_figs='figs'
fldr_res='group_results'
beta=0.6
i0=1.0
f_rate=0.1
graddiag=str(False)
M=10

plot_0=False # plot baseline for zero corruption

algs = [('BCORES', 'β-Cores', pal[0]), ('DShapley', 'Data Shapley', pal[1]), ('RAND', 'Random', pal[3])] #, ('RAND', 'Random', pal[3])] #
fldr_figs = 'figs'
if not os.path.exists(fldr_figs):
  os.mkdir(fldr_figs)

figs=[]
groups=[]
groupsll=[]
print('Plotting ' + dnm)
if dnm=='diabetes':
  fig = bkp.figure(y_axis_label='Predictive Accuracy',  x_axis_label='Number of Included Groups', plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig, '72pt', False, False)
  fig2 = bkp.figure(y_axis_label='Predictive Accuracy',  x_axis_label='Coreset Size', plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig2, '72pt', False, False)
  figs.append([fig, fig2])
else:
  fig = bkp.figure(y_axis_label='',  x_axis_label='Number of Included Groups', plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig, '72pt', False, False)
  fig2 = bkp.figure(y_axis_label='',  x_axis_label='Coreset Size', plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig2, '72pt', False, False)
  figs.append([fig, fig2])

for alg in algs:
  print('\n\n\n\n alg : ', alg)
  if alg[0]=='BCORES':
      trials = [fn for fn in os.listdir(fldr_res) if fn.startswith(dnm+'_'+alg[0]+'_'+str(f_rate)+'_'+str(beta))]
  else:
      trials = [fn for fn in os.listdir(fldr_res) if fn.startswith(dnm+'_'+alg[0]+'_'+str(f_rate)+'_')]
  if len(trials) == 0:
    fig.line([], [], color=alg[2], legend_label=alg[1], line_width=10); fig.patch([], [], color=alg[2], legend_label=alg[1], alpha=0.3)
    fig2.line([], [], color=alg[2], legend_label=alg[1], line_width=10); fig2.patch([], [], color=alg[2], legend_label=alg[1], alpha=0.3)
    continue
  accs = np.zeros((len(trials), M+1))
  #dem = np.zeros((len(trials), M+1))
  cszs = np.zeros((len(trials), M+1))
  numgroups = np.zeros((len(trials), M+1))
  for tridx, fn in enumerate(trials):
    f = open(os.path.join(fldr_res,fn), 'rb')
    res = pk.load(f) #(w, p, accs, pll)
    f.close()
    accs[tridx] = res[0]
    print(res[2][-1],'\n')
    if alg[0] == 'BCORES':
      groups+=res[2][-1]
      groupsll+=[res[2][-1]]

    #dem[tridx] = res[2]
    if alg[0]=='BCORES':
      cszs[tridx, :] = np.array([len(d) for d in res[1]])
      numgroups[tridx,:] = [len(set(r)) for r in res[2]]
    else:
      cszs[tridx, :] = np.array([len(d) for d in res[1][1:]])
      numgroups[tridx,:] = [len(set(r)) for r in res[2][1:]]


  csz50 = np.percentile(cszs, 50, axis=0)
  csz25 = np.percentile(cszs, 25, axis=0)
  csz75 = np.percentile(cszs, 75, axis=0)

  acc50 = np.percentile(accs, 50, axis=0)
  acc25 = np.percentile(accs, 25, axis=0)
  acc75 = np.percentile(accs, 75, axis=0)

  numg50 = np.percentile(numgroups, 50, axis=0)
  numg25 = np.percentile(numgroups, 25, axis=0)
  numg75 = np.percentile(numgroups, 75, axis=0)

  fig.line(numg50, acc50, color=alg[2], legend_label=alg[1], line_width=10)
  #fig.line(np.arange(acc25.shape[0]), acc25, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')
  #fig.line(np.arange(acc75.shape[0]), acc75, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')

  fig.patch(np.hstack((numg50, numg50[::-1])), np.hstack((acc75, acc25[::-1])), fill_color=alg[2], legend_label=alg[1], alpha=0.3)


  if alg[0] != 'PRIOR':
    fig2.line(csz50, acc50, color=alg[2], legend_label=alg[1], line_width=10)
    fig2.patch(np.hstack((csz50, csz50[::-1])), np.hstack((acc75, acc25[::-1])), fill_color=alg[2], legend_label=alg[1], alpha=0.3)
    fig2.legend.location='top_left'

  if plot_0:
    trials = [fn for fn in os.listdir(fldr_res) if fn.startswith(dnm+'_'+alg[0]+'_'+str(f_rate))]
    accs = np.zeros((len(trials), M))
    #dem = np.zeros((len(trials), M))
    cszs = np.zeros((len(trials), M))
    for tridx, fn in enumerate(trials):
      f = open(os.path.join(fldr_res,fn), 'rb')
      res = pk.load(f) #(w, p, accs, pll)
      f.close()
      accs[tridx] = res[0]
      #idcs[tridx] = res[1]
      #dem[tridx] = res[2]
      print('\n\n', res[2][-1], '\n\n\n')
      cszs[tridx, :] = np.array([len(d) for d in res[1]])



    csz50 = np.percentile(cszs, 50, axis=0)
    csz25 = np.percentile(cszs, 25, axis=0)
    csz75 = np.percentile(cszs, 75, axis=0)

    acc50 = np.percentile(accs, 50, axis=0)
    acc25 = np.percentile(accs, 25, axis=0)
    acc75 = np.percentile(accs, 75, axis=0)

    numg50 = np.percentile(numgroups, 50, axis=0)
    numg25 = np.percentile(numgroups, 25, axis=0)
    numg75 = np.percentile(numgroups, 75, axis=0)


    fig.line(np.arange(acc50.shape[0]), acc50, color=alg[2], legend_label=alg[1], line_width=10)
    fig.line(np.arange(acc25.shape[0]), acc25, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')
    fig.line(np.arange(acc75.shape[0]), acc75, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')

    if alg[0] != 'PRIOR':
      fig2.line(csz50, acc50, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')
      fig2.legend.location='top_left'


for f in [fig]:
  f.legend.location='top_right'
  f.legend.label_text_font_size= '80pt'
  f.legend.glyph_width=50
  f.legend.glyph_height=50
  f.legend.spacing=10
  f.legend.visible = False
for f in [fig2]:
  f.legend.location='top_right'
  f.legend.label_text_font_size= '80pt'
  f.legend.glyph_width=50
  f.legend.glyph_height=50
  f.legend.spacing=10
  f.legend.visible = False

fig2.add_layout(Title(text="F="+str(int(100*f_rate))+"%," + "  β="+str(beta), text_font_style="italic", text_font_size = "70px"), 'above')
fig2.add_layout(Title(text=dnmnm, align="center", text_font='helvetica', text_font_style='bold', text_font_size = "80px"), "above")

fig.add_layout(Title(text="F="+str(int(100*f_rate))+"%," + "  β="+str(beta), text_font_style="italic", text_font_size = "70px"), 'above')
fig.add_layout(Title(text=dnmnm, align="center", text_font='helvetica', text_font_style='bold', text_font_size = "80px"), "above")


fnm = (str(beta)+'_'+str(i0)+'_'+str(f_rate)+'_'+str(graddiag)).replace('.', '')
export_png(fig, filename=os.path.join(fldr_figs, 'group_'+dnm+fnm+"_ACCvsit.png"), height=1500, width=2000)
export_png(fig2, filename=os.path.join(fldr_figs, 'group_'+dnm+fnm+"_ACCvssz.png"), height=1500, width=2000)

np.savez('results/group_diagnostics_'+dnm+'_'+fnm, accs=accs)
#bkp.show(bkl.gridplot(figs))




import matplotlib.pyplot as plt
import collections
plt.rc('font', family='Helvetica')

flatten = lambda l: [item for sublist in l for item in sublist]

groupset = collections.Counter(groups)
print('\n\n\n', groups)
print('\n\n\n', groupset.keys(), len(groupset.keys()))
print('\n\n\n', groupset)
print(numgroups)
numgroups = min(len(groupset.keys()), 10)

fig, ax = plt.subplots(figsize=(16,10))
ax.tick_params(axis='both', which='major', labelsize=25)
ax.tick_params(axis='both', which='minor', labelsize=25)

image = np.zeros(shape=(numgroups, 5))
comgr = [g[0] for g in groupset.most_common()[:numgroups]]
ax.yaxis.set_ticks(list(range(numgroups))) #set the ticks to be a
ylabs = [str(int(g[0]*10))+'%, '+str(g[1:]) for g in comgr]
ylabs = [str(g)[:-1].replace("'", "").replace("(", "") for g in ylabs]

ax.set_ylabel('%Outliers, Age, Race, Gender', rotation='horizontal', fontsize=35, style='italic')
ax.yaxis.set_label_coords(-0.78,1.02)
ax.set_yticklabels(ylabs)
ax.set_xlabel('Trial', fontsize=35, style='italic')
ax.xaxis.set_ticks(list(range(5))) #set the ticks to be a
ax.set_xticklabels(['1','2','3','4','5']) #set the ticks to be a

for i, g in enumerate(comgr):
  image[i,:] = [0.15*(e+1)*(g in gr) for e, gr in enumerate(groupsll)]
image[image == 0] = 'nan'
print(image)


ax.imshow(image, cmap=plt.cm.jet, interpolation='nearest', alpha=0.6)
for edge, spine in ax.spines.items():
  spine.set_visible(False)

ax.set_xticks(np.arange(image.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(image.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
ax.tick_params(which="minor", bottom=False, left=False)
fig.tight_layout()

fig.savefig(os.path.join(fldr_figs, 'selected_groups.png'), format='png', dpi=300)
