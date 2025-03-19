import os
import argparse
from tqdm import tqdm
import pandas as pd
from time import time
import numpy as np
from LCC.continuous_lcc import ContinuousLCCMeasurer
from load_data.get_dsets import ImageStreamer, coffee_cream_sim
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--display-cluster-label-imgs',action='store_true')
parser.add_argument('--display-input-imgs',action='store_true')
parser.add_argument('--display-scattered-clusters',action='store_true')
parser.add_argument('-d','--dset',type=str,choices=['im','cifar','mnist','rand', 'bitrand','dtd','stripes','halves','fractal_imgs', 'naive-fluidsim', 'fluidsim', 'fluidsim-blurred'],default='stripes')
parser.add_argument('--downsample',type=int,default=-1,help='lower resolution of input images, -1 means no downsampling')
parser.add_argument('--given-fname',type=str,default='none')
parser.add_argument('--given-class-dir',type=str,default='none')
parser.add_argument('--include-mdl-abl',action='store_true',help='include ablation setting where mdl is not used and nc is set to 5 instead')
parser.add_argument('--n-cluster-inits',type=int,default=1,help='passed to the clustering model training')
parser.add_argument('--ncs-to-check',type=int,default=2,help='range of values of k to select from using mdl')
parser.add_argument('--no-resize',action='store_true',help="don't resize the input images")
parser.add_argument('--n-ims','-n',type=int,default=1)
parser.add_argument('--n-levels',type=int,default=4,help="how many scales to evaluate complexity at")
parser.add_argument('--nz',type=int,default=2, help='dimension to reduce to before clustering')
parser.add_argument('--select-randomly',action='store_true', help='shuffle the dataset before looping through')
parser.add_argument('--show-df',action='store_true', help='print the results as a pandas dataframe to stdout')
parser.add_argument('--cluster-model',type=str,choices=['kmeans','cmeans','gmm'],default='gmm')
parser.add_argument('--verbose','-v',action='store_true')
parser.add_argument('--print-times',action='store_true')
parser.add_argument('--suppress-all-prints',action='store_true')
parser.add_argument('--gaussian-noisify',action='store_true')
ARGS = parser.parse_args()

comp_meas = ContinuousLCCMeasurer(ncs_to_check=ARGS.ncs_to_check,
                               n_cluster_inits=ARGS.n_cluster_inits,
                               nz=ARGS.nz,
                               n_levels=ARGS.n_levels,
                               cluster_model=ARGS.cluster_model,
                               print_times=ARGS.print_times,
                               display_cluster_label_imgs=ARGS.display_cluster_label_imgs,
                               display_scattered_clusters=ARGS.display_scattered_clusters,
                               suppress_all_prints=ARGS.suppress_all_prints,
                               verbose=ARGS.verbose,
                               )

columns = ['model_cost', 'idx_cost', 'img_label', 'proc_time', 'lccscore', 'residuals', 'total']
results_df = pd.DataFrame(columns=columns)

img_start_times = []
img_times_real = []
labels = []
img_streamer = ImageStreamer(ARGS.dset,~ARGS.no_resize)
if ARGS.dset == 'naive-fluidsim':
    image_generator = coffee_cream_sim(ARGS.n_ims)
else:
    image_generator = img_streamer.stream_images(ARGS.n_ims,ARGS.downsample,ARGS.given_fname, ARGS.given_class_dir,ARGS.select_randomly)

for idx,(im,label) in enumerate(image_generator):
    img_label = label.split('_')[0] if ARGS.dset in ['im','dtd'] else label
    #print(idx, img_label)
    plt.axis('off')
    if ARGS.gaussian_noisify > 0:
        noise = np.random.randn(*im.shape)
        im += ARGS.gaussian_noisify*noise
        im = np.clip(im,0,1)

    img_start_times.append(time())
    if ARGS.display_input_imgs:
        plt.imshow(im);plt.show()
    im_normed = im
    if ARGS.include_mdl_abl:
        comp_meas.is_mdl_abl = True
        no_mdls, _, _ = comp_meas.interpret(im)
        comp_meas.is_mdl_abl = False
    else:
        no_mdls = [0]
    img_start_time = time()
    model_costs_by_level, idx_costs_by_level, lccs_by_level, residuals_by_level, total_by_level = comp_meas.interpret(im)
    results_df.loc[idx,'img_label'] = img_label
    results_df.loc[idx,'proc_time'] = time()-img_start_time
    results_df.loc[idx,'model_cost'] = sum(model_costs_by_level)
    results_df.loc[idx,'idx_cost'] = sum(idx_costs_by_level)
    results_df.loc[idx,'lccscore'] = sum(lccs_by_level)
    results_df.loc[idx,'residuals'] = sum(residuals_by_level)
    results_df.loc[idx,'total'] = sum(total_by_level)

    print(lccs_by_level, sum(lccs_by_level))
    #if idx%50 == 0:
        #plt.imshow(im)
        #plt.savefig(f'im{idx}.png')
        #os.system(f'/usr/bin/xdg-open im{idx}.png')
    labels.append(label)

results_df.index = results_df['img_label']
plt.xlabel('Time')
plt.ylabel('Complexity Score')
plt.plot(results_df['lccscore'])#; plt.show()
n_classes = len(set(results_df['img_label']))
n_per_class = len(results_df)/n_classes
preds = []
gts = []
class2idx = {x:i for i,x in enumerate(results_df['img_label'].unique())}
for i in range(n_classes):
    n_to_add = round(len(results_df)*(i+1)/n_classes) - len(preds)
    gts += [class2idx[x] for x in results_df['img_label']][len(preds):len(preds)+n_to_add]
    preds += [i]*n_to_add
    assert len(preds) == len(gts) == round(len(results_df)*(i+1)/n_classes)
oneseights = pd.concat([results_df.loc[results_df['img_label']==1], results_df.loc[results_df['img_label']==8]])
n1pred = len(oneseights)//2
n8pred = len(oneseights) - n1pred
results_df = results_df.drop('img_label',axis=1)
stds = results_df.std(axis=0)
means = results_df.mean(axis=0)
results_df.loc['stds'] = stds
results_df.loc['means'] = means
os.makedirs('results/images', exist_ok=True)
results_df.to_csv(f'results/images/{ARGS.dset}_results.csv')
results_df.to_csv(f'results/images/{ARGS.dset}_results-{ARGS.n_ims}ims.csv')

if ARGS.show_df:
    print(results_df)
print(results_df.loc['means'])
