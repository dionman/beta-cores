import bokeh.plotting as bkp
from bokeh.io import export_png, export_svgs
import numpy as np
import sys, os
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *

n_trials=int(sys.argv[1])
plot_every=int(sys.argv[2])
fldr_figs=sys.argv[3]
prfx=sys.argv[4]
f_rate = str(sys.argv[5])

plot_reverse_kl = True
trials = np.arange(1, n_trials)
nms = [('BCORES', 'Beta-CORES', pal[0]), ('BPSVI', 'PSVI', pal[1]), ('SVI', 'SparseVI', pal[2]), ('RAND', 'Uniform', pal[3]), ('GIGAO','GIGA (Optimal)', pal[4]), ('GIGAR','GIGA (Realistic)', pal[5])]
pcsts = ['PSVI'] # pseudocoreset nameslist

#plot the KL figure
fig = bkp.figure(y_axis_type='log', plot_width=850, plot_height=850, x_axis_label='Coreset Size',
       y_axis_label=('Reverse KL' if plot_reverse_kl else 'Forward KL'), toolbar_location=None )
preprocess_plot(fig, '32pt', False, True)

for i, nm in enumerate(nms):
  kl = []
  sz = []
  for tr in trials:
    numTuple = (prfx, f_rate, nm[0], str(tr))
    if nm[0]=='BCORES':
      x_, mu0_, Sig0_, Sig_, mup_, Sigp_, w_, p_, muw_, Sigw_, rklw_, fklw_, beta_ = np.load(os.path.join(prfx, '_'.join(numTuple)+'.pk'), allow_pickle=True)
    else:
      x_, mu0_, Sig0_, Sig_, mup_, Sigp_, w_, p_, muw_, Sigw_, rklw_, fklw_ = np.load(os.path.join(prfx, '_'.join(numTuple)+'.pk'), allow_pickle=True)
    if plot_reverse_kl:
      kl.append(rklw_[::plot_every])
    else:
      kl.append(fklw_[::plot_every])
    if nm[0] in pcsts:
      sz.append(range(len(w_))[::plot_every])
    else:
      sz.append([np.count_nonzero(a) for a in w_[::plot_every]])
  x = np.percentile(sz, 50, axis=0)
  # HACK FOR REPRODUCING COLOURS OF NEURIPS PAPER:
  #assign PSVI to last colour of the pallete in order to maintain colouring of previous papers for baselines
  fig.line(x, np.percentile(kl, 50, axis=0), color=nm[-1], line_width=5, legend_label=nm[1])
  fig.patch(x = np.hstack((x, x[::-1])), y = np.hstack((np.percentile(kl, 75, axis=0), np.percentile(kl, 25, axis=0)[::-1])), color=nm[-1], fill_alpha=0.4, legend_label=nm[1])

postprocess_plot(fig, '30pt', location='bottom_right', glyph_width=40)
fig.legend.background_fill_alpha=0.
fig.legend.border_line_alpha=0.
fig.legend.visible = True

bkp.show(fig)
if not os.path.exists(fldr_figs):
  os.mkdir(fldr_figs)
export_png(fig, filename=os.path.join(fldr_figs, "KLDvsCstSize.png"), height=1500, width=1500)
