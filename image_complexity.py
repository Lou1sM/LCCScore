import os
import argparse
import pandas as pd
from time import time
from LCC.continuous_lcc import ContinuousLCCMeasurer
from load_data.get_dsets import ImageStreamer


parser = argparse.ArgumentParser()
parser.add_argument('-d','--dset',type=str,choices=['im','cifar','mnist','rand', 'bitrand','dtd','stripes','halves','fractal_imgs'],default='stripes')
parser.add_argument('--downsample',type=int,default=-1,help='lower resolution of input images, -1 means no downsampling')
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
ARGS = parser.parse_args()

comp_meas = ContinuousLCCMeasurer(ncs_to_check=ARGS.ncs_to_check,
                               n_cluster_inits=ARGS.n_cluster_inits,
                               nz=ARGS.nz,
                               n_levels=ARGS.n_levels,
                               cluster_model=ARGS.cluster_model,
                               verbose=ARGS.verbose,
                               )

columns = ['model_cost', 'idx_cost', 'img_label', 'lccscore', 'residuals', 'total']
results_df = pd.DataFrame(columns=columns)

img_streamer = ImageStreamer(ARGS.dset,~ARGS.no_resize)
image_generator = img_streamer.stream_images(ARGS.n_ims, ARGS.downsample, select_randomly=ARGS.select_randomly)

for idx,(im,label) in enumerate(image_generator):
    img_label = label.split('_')[0] if ARGS.dset in ['im','dtd'] else label

    img_start_time = time()
    model_costs_by_level, idx_costs_by_level, lccs_by_level, residuals_by_level, total_by_level = comp_meas.interpret(im)
    results_df.loc[idx,'img_label'] = img_label
    results_df.loc[idx,'proc_time'] = time()-img_start_time
    results_df.loc[idx,'model_cost'] = sum(model_costs_by_level)
    results_df.loc[idx,'idx_cost'] = sum(idx_costs_by_level)
    results_df.loc[idx,'lccscore'] = sum(lccs_by_level)
    results_df.loc[idx,'residuals'] = sum(residuals_by_level)
    results_df.loc[idx,'total'] = sum(total_by_level)

results_df.index = results_df['img_label']
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
