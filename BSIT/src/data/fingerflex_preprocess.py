import os

from fingerflex_utils import compute_xr_ecog_ff


lp = '../../../scratch/yuhxie/fingerflex/data/'
sp = 'src/data/fingerflex/'
tlims = [-2,2] # seconds
tlims_handpos = [0,4] # seconds
filt_freqs = [4, 250]  # [1,None] # Hz (low, high cutoffs)
sfreq_new = 250 # 250 # Hz
# 'mv' cue variable does not change (stim variable looks fine)
out_sbj_d = {'bp':'S01', 'cc':'S02', 'ht':'S03', 'jc':'S04', 'jp':'S05',
             'mv':'S06', 'wc':'S07', 'wm':'S08', 'zt':'S09'}


if not os.path.exists(sp):
    os.mkdir(sp)
if not os.path.exists(sp+'/pose/'):
    os.mkdir(sp+'/pose/')

for sbj_id in ['bp', 'cc', 'ht', 'jc', 'jp', 'wc', 'wm', 'zt']:
    compute_xr_ecog_ff(sbj_id, lp, sp, tlims, tlims_handpos,
                       filt_freqs, sfreq_new, out_sbj_d)
