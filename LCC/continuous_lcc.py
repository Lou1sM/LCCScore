from time import time
from matplotlib.colors import BASE_COLORS
from dl_utils.misc import scatter_clusters
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from numpy import log2
from .utils import opt_cost_from_discrete_seq

np.random.seed(0)

PALETTE = list(BASE_COLORS.values()) + [(0,0.5,1),(1,0.5,0)]
class ContinuousLCCMeasurer():
    def __init__(self, ncs_to_check, n_levels, cluster_model, n_cluster_inits=1, nz=2,
                    display_cluster_label_imgs=False,suppress_all_prints=False,
                    verbose=False, display_scattered_clusters=False, print_times=False,
                    cluster_verbose=False, is_mdl_abl=False):

        self.verbose = verbose
        self.cluster_verbose = cluster_verbose
        self.suppress_all_prints = suppress_all_prints
        self.n_levels = n_levels
        self.print_times = print_times
        self.n_cluster_inits = n_cluster_inits
        self.display_cluster_label_imgs = display_cluster_label_imgs
        self.display_scattered_clusters = display_scattered_clusters
        self.ncs_to_check = ncs_to_check
        self.thresh_nz = nz
        assert cluster_model in ['kmeans','cmeans','gmm']
        self.cluster_model = cluster_model
        self.is_mdl_abl = is_mdl_abl

    def interpret(self, given_x, **kwargs):
        orig_vals = {}
        for k,v in kwargs.items():
            orig_vals[k] = getattr(self, k)
            setattr(self, k, v)

        self.is_large_im = np.prod(given_x.shape) > 1e4
        x = np.copy(given_x)
        all_lccss = []
        all_model_costs = []
        all_idxs_costs = []
        all_residual_costs = []
        self.set_smallest_increment(x)
        for layer_being_processed in range(self.n_levels):
            self.layer_being_processed = layer_being_processed
            cluster_start_time = time()
            self.mdl_cluster(x)
            all_lccss.append(self.lccs)
            all_model_costs.append(self.best_model_cost)
            all_idxs_costs.append(self.best_idxs_cost)
            all_residual_costs.append(self.best_residuals_cost)
            if self.display_cluster_label_imgs:
                self.viz_cluster_labels()
            if self.best_nc == 0:
                pad_len = self.n_levels - len(all_lccss)
                all_lccss += [0]*pad_len
                all_residual_costs += [self.best_residuals_cost]*pad_len
                break
            if self.print_times:
                print(f'mdl_cluster time: {time()-cluster_start_time:.2f}')
            if self.best_nc == 1:
                x = np.zeros_like(x)
            else:
                cluster_labels_non_outliers = self.best_cluster_labels.flatten()
                cluster_labels_non_outliers[self.outliers] = -1
                cluster_labels_non_outliers = cluster_labels_non_outliers.reshape(self.best_cluster_labels.shape)
                bool_ims_by_c = [(cluster_labels_non_outliers==c)
                                for c in np.unique(self.best_cluster_labels)]
                one_hot_im = np.stack(bool_ims_by_c, axis=2)
                if self.is_large_im:
                    patch_size = 4*(2**layer_being_processed)
                else:
                    patch_size = 2*(layer_being_processed+1)
                if self.verbose:
                    print(f'patch_size: {patch_size}')
                c_idx_patches = combine_patches(one_hot_im, patch_size)
                x = c_idx_patches
        for k,v in orig_vals.items():
            setattr(self, k, v)
        all_totals = [m+r for m,r in zip(all_lccss, all_residual_costs)]
        return all_model_costs, all_idxs_costs, all_lccss, all_residual_costs, all_totals

    def set_smallest_increment(self, x):
        self.bit_prec = 32 # cuz this is what it'll be for rand so keep it the same for everything

    def mdl_cluster(self, x_as_img, fixed_nc=-1):
        x = x_as_img.reshape(-1, x_as_img.shape[-1])
        assert x.ndim == 2
        N, nz = x.shape
        if self.verbose:
            print('number pixel-likes to cluster:', N)
            print('number different pixel-likes:', nz)
        if nz > max(self.thresh_nz, 50):
            breakpoint()
            x = PCA(50).fit_transform(x)
        if nz > max(self.thresh_nz, 3):
            dim_red_start_time = time()
            x = PCA(self.thresh_nz).fit_transform(x)
            if self.print_times:
                print(f'dim red time: {time()-dim_red_start_time:.2f}')
        N, nz = x.shape
        self.nz = nz
        data_range = x.max() - x.min()
        self.len_of_each_cluster = 2 * nz * (log2(data_range) + self.bit_prec) # Float precision
        self.len_of_outlier = nz * (log2(data_range) + self.bit_prec)
        #self.best_cost = N*self.len_of_outlier # Could call everything an outlier and have no clusters
        self.best_cost = N*self.len_of_outlier
        self.best_model_cost = 0
        self.best_nc = 0
        self.best_outliers = np.zeros(x.shape[0]).astype(bool)
        self.best_outliers_cost = 0
        self.best_outlier_mask_cost = 0
        self.best_idxs_cost = 0
        self.best_residuals_cost = self.best_cost
        if len(np.unique(x))==1:
            self.best_nc = 1
            self.best_cost = 0
            self.best_residuals_cost = 0
        else:
            nc_start_times = []
            ncs_to_check = [5] if self.is_mdl_abl else range(1, self.ncs_to_check+1)
            self.best_means = x.mean(axis=0)
            if self.verbose:
                print(( f'0 {self.best_cost/N:.3f}\tMod: {self.best_model_cost/N:.3f}\t'
                        f'Err: {self.best_residuals/N:.3f}\t'
                        f'Idxs: {self.best_idxs_cost/N:.3f}\t'
                        f'OMask: {self.best_outlier_mask_cost/N:.3f}\t'
                        f'O: {self.best_outliers.sum()} {self.best_outliers_cost/N:.3f}\t'))
            for nc in ncs_to_check:
                nc_start_times.append(time())
                found_nc = self.cluster(x, nc)
                if self.verbose:
                    print(( f'{nc} {self.dl/N:.3f}\tMod: {self.model_cost/N:.3f}\t'
                            f'Err: {self.cluster_residuals/N:.3f}\t'
                            f'Idxs: {self.idxs_cost/N:.3f}\t'
                            f'OMask: {self.best_outlier_mask_cost/N:.3f}\t'
                            f'O: {self.outliers.sum()} {self.outliers_cost/N:.3f}\t'))
                if self.dl < self.best_cost or self.is_mdl_abl:
                    self.best_cost = self.dl
                    self.best_model_cost = self.model_cost
                    self.best_idxs_cost = self.idxs_cost
                    self.best_residuals_cost = self.cluster_residuals + self.outliers_cost
                    self.best_nc = nc
                    self.best_outliers = self.outliers
                    self.best_outliers_cost = self.outliers_cost
                    self.best_outlier_mask_cost = self.outlier_mask_cost
                    self.best_cluster_labels = self.cluster_labels.reshape(*x_as_img.shape[:-1])
                    if self.cluster_model=='gmm':
                        self.best_means = self.model.means_
                        self.best_covs = self.model.covariances_
                    else:
                        self.best_means = self.model.cluster_centers_
                        self.best_covs = np.eye(len(self.best_means))
                if found_nc == nc-1 and not self.suppress_all_prints:
                    print(f"only found {nc-1} clusters when looking for {nc}, terminating here")
                    break
            if self.print_times:
                nc_times = [nc_start_times[i+1] - ncs for i, ncs in enumerate(nc_start_times[:-1])]
                tot_c_time = f' tot: {nc_start_times[-1] - nc_start_times[0]:.2f}'
                print(' '.join([f'{i}: {s:.2f}' for i, s in enumerate(nc_times)]) + tot_c_time)
            if self.verbose: print(f'found {self.best_nc} clusters')
        if np.isinf(self.best_residuals_cost):
            breakpoint()
        self.lccs = self.best_model_cost + self.best_idxs_cost
        x_copy = np.copy(x)
        x_copy[self.best_outliers] = -1
        self.full_patch_info = opt_cost_from_discrete_seq(x_copy)
        self.full_patch_residual = opt_cost_from_discrete_seq(x[self.best_outliers])

    def cluster(self, x, nc):
        if self.cluster_model == 'kmeans':
            self.model_cost = 0.5 * nc * self.len_of_each_cluster
            found_nc = nc
            self.model = KMeans(nc)
            self.cluster_labels = self.model.fit_predict(x)
            sizes_of_each_cluster = [np.zeros(x.shape[1]) if (self.cluster_labels==i).sum()==0 else x[self.cluster_labels==i].max(axis=0)-x[self.cluster_labels==i].min(axis=0) for i in range(nc)]
            neg_log_prob_per_cluster = np.array([log2(dr).sum() for dr in sizes_of_each_cluster])
            neg_log_probs = neg_log_prob_per_cluster[self.cluster_labels]
        else:
            self.model = GMM(nc, n_init=self.n_cluster_inits, covariance_type='diag')
            while True:
                try:
                    self.cluster_labels = self.model.fit_predict(x)
                    break
                except ValueError:
                    if not self.suppress_all_prints:
                        print(f'failed to cluster with {nc} components, and reg_covar {self.model.reg_covar}')
                    self.model.reg_covar *= 10
                    if self.model.reg_covar > 10:
                        breakpoint()
                    if not self.suppress_all_prints:
                        print(f'trying again with reg_covar {self.model.reg_covar}')
            found_nc = len(np.unique(self.cluster_labels))
            if nc > 1 and self.display_scattered_clusters:
                scatter_clusters(x, self.best_cluster_labels.flatten(), show=True)
            self.model_cost = nc*(self.len_of_each_cluster)
            new_model_scores = -self.model._estimate_log_prob(x)[np.arange(len(x)), self.cluster_labels] * log2(np.e)
            neg_log_probs = new_model_scores + self.nz*self.bit_prec
        self.outliers = neg_log_probs > self.len_of_outlier
        self.outliers_cost = (self.len_of_outlier * self.outliers).sum()
        self.idxs_cost = opt_cost_from_discrete_seq(self.cluster_labels[~self.outliers])
        self.outlier_mask_cost = opt_cost_from_discrete_seq(self.outliers)
        self.cluster_residuals = (neg_log_probs * ~self.outliers).sum()
        self.dl = self.cluster_residuals + self.outliers_cost + self.idxs_cost + self.model_cost + self.outlier_mask_cost
        return found_nc

    def viz_cluster_labels(self):
        pallete = PALETTE[:self.best_nc]
        to_display = np.array(pallete)[self.best_cluster_labels]
        to_display = np.resize(to_display,(*self.best_cluster_labels.shape,3))
        plt.imshow(to_display)
        plt.savefig('segmentation_by_clusters.png', bbox_inches=0)
        plt.show()

    def viz_reconstruction(self):
        to_display = np.tile(np.expand_dims(np.zeros_like(self.best_cluster_labels), 2), (1,1,3)).astype(float)
        nr, nc = self.best_cluster_labels.shape
        for i in range(self.best_nc):
            sample = np.random.normal(loc=self.best_means[i], scale=np.sqrt(self.best_covs[i]), size=(nr, nc ,3))
            to_display[self.best_cluster_labels==i] = sample[self.best_cluster_labels==i]
        outliers_mask = self.outliers.reshape(nr, nc)
        to_display[outliers_mask] = np.random.rand(*to_display.shape)[outliers_mask]
        plt.imshow(to_display)
        plt.savefig('lossy_reconstructed_img.png', bbox_inches=0)
        plt.show()

def viz_proc_im(x):
    plt.imshow(x.sum(axis=2)); plt.show()

def combine_patches(a, ps, n_im_dim=2):
    n_rows, n_cols = a.shape[:2]
    comb_func = lambda x: sum([z.astype(float) for z in x])
    if n_cols > ps:
        to_add_horizontally = [a[:,i:i-ps] for i in range(ps)]
        a = comb_func(to_add_horizontally)
    if n_rows > ps and n_im_dim == 2:
        to_add_vertically = [a[i:i-ps] for i in range(ps)]
        a = comb_func(to_add_vertically)
    return a
