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

dnm=sys.argv[1]
i0=sys.argv[2]
f_rate=sys.argv[3]
beta=sys.argv[4]
if dnm=='boston':
  dnmnm='HOUSING'
elif dnm=='year':
  dnmnm='SONGS'
fldr_figs='figs'
fldr_res='results'

algs = [('BCORES', 'β-Cores', pal[0]), ('SVI', 'SparseVI', pal[4]), ('RAND', 'Uniform', pal[3])] #
fldr_figs = 'figs'
if not os.path.exists(fldr_figs):
  os.mkdir(fldr_figs)

figs=[]
print('Plotting ' + dnm)
if dnm=='adult':
  fig = bkp.figure(y_axis_label= 'RMSE',  x_axis_label='# Iterations', plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig, '72pt', False, False)
  fig2 = bkp.figure(y_axis_label='RMSE',  x_axis_label='Coreset Size', plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig2, '72pt', False, False)
  fig3 = bkp.figure(y_axis_label='Negative Predictive Log-Likelihood',  x_axis_label='# Iterations', plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig3, '72pt', False, False)
  fig4 = bkp.figure(y_axis_label='Predictive LogLik',  x_axis_label='Coreset Size',   plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig4, '72pt', False, False)
  figs.append([fig, fig2, fig3, fig4])
else:
  fig = bkp.figure(y_axis_label='Test RMSE',  x_axis_label='# Iterations for Batch Selection', plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig, '72pt', False, False)
  fig2 = bkp.figure(y_axis_label='Test RMSE',  x_axis_label='Coreset Size', plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig2, '72pt', False, False)
  fig3 = bkp.figure(y_axis_label='',  x_axis_label='# Iterations', plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig3, '72pt', False, False)
  fig4 = bkp.figure(y_axis_label='',  x_axis_label='Coreset Size',   plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig4, '72pt', False, False)
  figs.append([fig, fig2, fig3, fig4])
M=21
for alg in algs:
  trials = [fn for fn in os.listdir(fldr_res) if fn.startswith('results_'+dnm+'_'+alg[0]+'_frate_'+str(f_rate)+'_beta_'+str(beta)+'_i0_'+str(i0))]
  if len(trials) == 0:
    fig.line([], [], color=alg[2], legend_label=alg[1], line_width=10); fig.patch([], [], color=alg[2], legend_label=alg[1], alpha=0.3)
    fig2.line([], [], color=alg[2], legend_label=alg[1], line_width=10); fig2.patch([], [], color=alg[2], legend_label=alg[1], alpha=0.3)
    fig3.line([], [], color=alg[2], legend_label=alg[1], line_width=10); fig3.patch([], [], color=alg[2], legend_label=alg[1], alpha=0.3)
    fig4.line([], [], color=alg[2], legend_label=alg[1], line_width=10); fig4.patch([], [], color=alg[2], legend_label=alg[1], alpha=0.3)
    continue
  rmses = np.zeros((len(trials), M))
  plls = np.zeros((len(trials), M))
  cszs = np.zeros((len(trials), M))
  for tridx, fn in enumerate(trials):
    f = open(os.path.join(fldr_res,fn), 'rb')
    res = pk.load(f) #(w, p, rmses, pll)
    f.close()
    wts = res[0]
    pts = res[1]
    rmses[tridx] = res[2]
    plls[tridx]  = -res[3]
    cszs[tridx, :] = np.array([len(w) for w in wts])

  csz50 = np.percentile(cszs, 50, axis=0)
  csz25 = np.percentile(cszs, 25, axis=0)
  csz75 = np.percentile(cszs, 75, axis=0)

  rmse50 = np.percentile(rmses, 50, axis=0)
  rmse25 = np.percentile(rmses, 25, axis=0)
  rmse75 = np.percentile(rmses, 75, axis=0)

  pll50 = np.percentile(plls, 50, axis=0)
  pll25 = np.percentile(plls, 25, axis=0)
  pll75 = np.percentile(plls, 75, axis=0)


  fig.line(np.arange(rmse50.shape[0]), rmse50, color=alg[2], legend_label=alg[1], line_width=10)
  #fig.line(np.arange(rmse25.shape[0]), rmse25, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')
  #fig.line(np.arange(rmse75.shape[0]), rmse75, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')
  fig.patch(np.hstack((np.arange(rmse75.shape[0]), np.arange(rmse75.shape[0])[::-1])), np.hstack((rmse75, rmse25[::-1])), fill_color=alg[2], legend_label=alg[1], alpha=0.1)
  fig.patch(np.hstack((np.arange(rmse75.shape[0]), np.arange(rmse75.shape[0])[::-1])), np.hstack((rmse75, rmse25[::-1])), fill_color=alg[2], legend_label=alg[1], alpha=0.1)


  fig3.line(np.arange(rmse50.shape[0]), pll50, color=alg[2], legend_label=alg[1], line_width=10)
  fig3.line(np.arange(rmse25.shape[0]), pll25, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')
  fig3.line(np.arange(rmse75.shape[0]), pll75, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')

  if alg[0] != 'PRIOR':
    fig2.line(csz50, rmse50, color=alg[2], legend_label=alg[1], line_width=10)
    fig2.patch(np.hstack((csz50, csz50[::-1])), np.hstack((rmse75, rmse25[::-1])), fill_color=alg[2], legend_label=alg[1], alpha=0.3)
    fig2.legend.location='top_left'
    fig4.line(csz50, pll50, color=alg[2], legend_label=alg[1], line_width=10)
    fig4.patch(np.hstack((csz50, csz50[::-1])), np.hstack((pll75, pll25[::-1])), fill_color=alg[2], legend_label=alg[1], alpha=0.3)
    fig4.legend.location='top_left'


for f in [fig, fig3]:
  f.legend.location='top_right'
  f.legend.label_text_font_size= '80pt'
  f.legend.glyph_width=50
  f.legend.glyph_height=50
  f.legend.spacing=10
  f.legend.visible = ((dnm=='boston') and (f_rate=="30"))
for f in [fig2, fig4]:
  f.legend.location='top_right'
  f.legend.label_text_font_size= '80pt'
  f.legend.glyph_width=50
  f.legend.glyph_height=50
  f.legend.spacing=10
  f.legend.visible = ((dnm=='boston') and (f_rate=="30"))

fig2.add_layout(Title(text="F="+str(f_rate)+"%," + "  β="+str(beta), text_font_style="italic", text_font_size = "70px"), 'above')
fig2.add_layout(Title(text=dnmnm, align="center", text_font='helvetica', text_font_style='bold', text_font_size = "80px"), "above")


fnm = (str(beta)+'_'+str(i0)+'_'+str(f_rate)).replace('.', '')
export_png(fig, filename=os.path.join(fldr_figs, dnm+fnm+"_RMSEvsvsit.png"), height=1500, width=2000)
export_png(fig2, filename=os.path.join(fldr_figs, dnm+fnm+"_RMSEvssz.png"), height=1500, width=2000)
