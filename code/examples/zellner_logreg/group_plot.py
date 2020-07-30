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
beta=0.9
i0=1.0
f_rate=0
graddiag=str(False)
M=11

plot_0=False # plot baseline for zero corruption

algs = [('BCORES', 'β-Cores', pal[0]), ('DShapley', 'Data Shapley', pal[1]), ('RAND', 'Uniform', pal[3])] #
fldr_figs = 'figs'
if not os.path.exists(fldr_figs):
  os.mkdir(fldr_figs)

figs=[]
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
  trials = [fn for fn in os.listdir(fldr_res) if fn.startswith(dnm+'_'+alg[0]+'_'+str(f_rate))]
  print(trials)
  if len(trials) == 0:
    fig.line([], [], color=alg[2], legend_label=alg[1], line_width=10); fig.patch([], [], color=alg[2], legend_label=alg[1], alpha=0.3)
    fig2.line([], [], color=alg[2], legend_label=alg[1], line_width=10); fig2.patch([], [], color=alg[2], legend_label=alg[1], alpha=0.3)
    continue
  accs = np.zeros((len(trials), M))
  dem = np.zeros((len(trials), M))
  cszs = np.zeros((len(trials), M))
  for tridx, fn in enumerate(trials):
    f = open(os.path.join(fldr_res,fn), 'rb')
    res = pk.load(f) #(w, p, accs, pll)
    f.close()
    accs[tridx] = res[0]
    idcs[tridx] = res[1]
    dem[tridx] = res[2]
    cszs[tridx, :] = np.array([len(d) for d in idcs[tridx]])

  csz50 = np.percentile(cszs, 50, axis=0)
  csz25 = np.percentile(cszs, 25, axis=0)
  csz75 = np.percentile(cszs, 75, axis=0)

  acc50 = np.percentile(accs, 50, axis=0)
  acc25 = np.percentile(accs, 25, axis=0)
  acc75 = np.percentile(accs, 75, axis=0)

  fig.line(np.arange(acc50.shape[0]), acc50, color=alg[2], legend_label=alg[1], line_width=10)
  fig.line(np.arange(acc25.shape[0]), acc25, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')
  fig.line(np.arange(acc75.shape[0]), acc75, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')

  if alg[0] != 'PRIOR':
    fig2.line(csz50, acc50, color=alg[2], legend_label=alg[1], line_width=10)
    fig2.patch(np.hstack((csz50, csz50[::-1])), np.hstack((acc75, acc25[::-1])), fill_color=alg[2], legend_label=alg[1], alpha=0.3)
    fig2.legend.location='top_left'

  if plot_0:
    trials = [fn for fn in os.listdir(fldr_res) if fn.startswith(dnm+'_'+alg[0]+'_'+str(f_rate))]
    accs = np.zeros((len(trials), M))
    dem = np.zeros((len(trials), M))
    cszs = np.zeros((len(trials), M))
    for tridx, fn in enumerate(trials):
      f = open(os.path.join(fldr_res,fn), 'rb')
      res = pk.load(f) #(w, p, accs, pll)
      f.close()
      accs[tridx] = res[0]
      idcs[tridx] = res[1]
      dem[tridx] = res[2]
      cszs[tridx, :] = np.array([len(d) for d in idcs[tridx]])

    csz50 = np.percentile(cszs, 50, axis=0)
    csz25 = np.percentile(cszs, 25, axis=0)
    csz75 = np.percentile(cszs, 75, axis=0)

    acc50 = np.percentile(accs, 50, axis=0)
    acc25 = np.percentile(accs, 25, axis=0)
    acc75 = np.percentile(accs, 75, axis=0)


    fig.line(np.arange(acc50.shape[0]), acc50, color=alg[2], legend_label=alg[1], line_width=10)
    fig.line(np.arange(acc25.shape[0]), acc25, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')
    fig.line(np.arange(acc75.shape[0]), acc75, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')

    if alg[0] != 'PRIOR':
      fig2.line(csz50, acc50, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')
      fig2.legend.location='top_left'


for f in [fig]:
  f.legend.location='top_left'
  f.legend.label_text_font_size= '80pt'
  f.legend.glyph_width=50
  f.legend.glyph_height=50
  f.legend.spacing=10
  f.legend.visible = (alg[0]=='adult')
for f in [fig2]:
  f.legend.location='bottom_right'
  f.legend.label_text_font_size= '80pt'
  f.legend.glyph_width=50
  f.legend.glyph_height=50
  f.legend.spacing=10
  f.legend.visible = (alg[0]=='adult')

fig2.add_layout(Title(text="F="+str(f_rate)+"%," + "  β="+str(beta), text_font_style="italic", text_font_size = "70px"), 'above')
fig2.add_layout(Title(text=dnmnm, align="center", text_font='helvetica', text_font_style='bold', text_font_size = "80px"), "above")


fnm = (str(beta)+'_'+str(i0)+'_'+str(f_rate)+'_'+str(graddiag)).replace('.', '')
export_png(fig, filename=os.path.join(fldr_figs, 'group_'+dnm+fnm+"_ACCvsit.png"), height=1500, width=2000)
export_png(fig2, filename=os.path.join(fldr_figs, 'group_'+dnm+fnm+"_ACCvssz.png"), height=1500, width=2000)

np.savez('results/group_diagnostics_'+dnm+'_'+fnm, accs=accs, plls=plls)
#bkp.show(bkl.gridplot(figs))
