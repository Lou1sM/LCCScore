from LCC.continuous_lcc import ContinuousLCCMeasurer
from PIL import Image
import numpy as np

comp_meas = ContinuousLCCMeasurer(ncs_to_check=5, n_levels=3, cluster_model='gmm')
im = Image.open('data/library3-big.jpg')
im = im.resize([224,224]) # it can run on the full image but slow and gives almost identical results
im = np.array(im)
model_costs_by_level, idx_costs_by_level, lccs_by_level, residuals_by_level, total_by_level = comp_meas.interpret(im)
lcc_score = sum(lccs_by_level)
print(lcc_score)

