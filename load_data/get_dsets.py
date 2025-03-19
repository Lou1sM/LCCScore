import os
import numpy as np
import pickle
from natsort import natsorted
import sys
from PIL import Image
from os import listdir
from os.path import join
import struct
from concurrent.futures import ThreadPoolExecutor
from sklearn.datasets import fetch_20newsgroups
from .create_simple_imgs import create_simple_img


imagenette_synset_dict = {
    'n01440764': 'tench',
    'n02102040': 'spaniel',
    'n02979186': 'cassette player',
    'n03000684': 'chain saw',
    'n03028079': 'church',
    'n03394916': 'French horn',
    'n03417042': 'garbage truck',
    'n03425413': 'gas pump',
    'n03445777': 'golf ball',
    'n03888257': 'parachute'
    }

def load_ng_feats():
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    text_feats = np.concatenate([np.load(f'data/ng20-featvecs/{fn}') for fn in natsorted(os.listdir('data/ng20-featvecs'))])
    labels = newsgroups.target
    return text_feats, labels

def load_im_feats():
    im_feats = []
    class_labels = []
    for class_dir in natsorted(os.listdir('data/im-featvecs')):
        for fn in natsorted(os.listdir(f'data/im-featvecs/{class_dir}')):
            im_feats.append(np.load(f'data/im-featvecs/{class_dir}/{fn}'))
            class_labels.append(class_dir)
    im_feats = np.stack(im_feats)
    class2num = {cl:i for i,cl in enumerate(set(class_labels))}
    labels = np.array([class2num[x] for x in class_labels])
    return im_feats, labels

def load_all_in_tree(im_dir):
    image_paths = [os.path.join(dirpath, fn) for dirpath, dirnames, filenames in os.walk(im_dir) for fn in filenames]
    image_paths = [fp for fp in image_paths if fp.endswith('JPEG')]
    with ThreadPoolExecutor(max_workers=4) as executor:
        loaded_images = list(executor.map(load_image, image_paths))
    return np.stack(loaded_images), image_paths

def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224), Image.LANCZOS)
    return np.array(img)

def load_mnist(split):
    split = 't10k' if split=='test' else 'train'
    with open(f'data/mnist/{split}-images-idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        imgs = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        imgs = imgs.reshape((size, nrows, ncols))
    with open(f'data/mnist/{split}-labels-idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    return imgs, labels

def load_rand(dset,resize=False):
    if dset=='imagenette':
        dset_dir = 'data/imagenette2/val'
        class_dir = np.random.choice(listdir(dset_dir))
        class_dir_path = join(dset_dir,class_dir)
    elif dset=='dtd':
        class_dir_path = 'data/dtd/suitable'
    fname = np.random.choice(listdir(class_dir_path))
    fpath = join(class_dir_path,fname)
    print(fname)
    return load_fpath(fpath,resize)

def load_fpath(fpath,resize,downsample):
    im = Image.open(fpath)
    if resize:
        h,w = im.size[:2]
        aspect_ratio = h/w
        new_h = (224*224*aspect_ratio)**0.5
        new_w = 224*224/new_h
        new_h_int = round(new_h)
        new_w_int = round(new_w)
        max_possible_error = (new_h_int + new_w_int) / 2
        if not (new_h_int*new_w_int - 224*224) < max_possible_error:
            breakpoint()
        if downsample != -1:
            im = im.resize((downsample,downsample))
        im = im.resize((new_h_int,new_w_int))
    im = np.array(im)
    if im.ndim==2:
        im = np.tile(np.expand_dims(im,2),(1,1,3))
    return im

def generate_non_torch_im(dset,resize,subsample):
    if dset=='imagenette':
        dset_dir = 'data/imagenette2/val'
    elif dset=='dtd':
        dset_dir = 'data/dtd/suitable'
    for i in range(subsample):
        if dset=='imagenette':
            num_classes = len(listdir(dset_dir))
            class_dir = listdir(dset_dir)[i%num_classes]
            idx_within_class = i//num_classes
            fname = listdir(join(dset_dir,class_dir))[idx_within_class]
            fpath = join(dset_dir,class_dir,fname)
        elif dset=='dtd':
            try:
                fname = listdir(dset_dir)[i]
            except IndexError:
                print(f"have run out of images, at image number {i}")
                sys.exit()
            fpath = join(dset_dir,fname)
        yield load_fpath(fpath,resize), fpath

def switch_rand_pos(img, switch_pos_x, switch_pos_y):
    if switch_pos_x is None:
        switch_pos_x = np.random.choice(img.shape[0]-1)
    if switch_pos_y is None:
        switch_pos_y = np.random.choice(img.shape[1])
    tmp = img[switch_pos_x+1, switch_pos_y, 0]
    img[switch_pos_x+1, switch_pos_y, 0] = img[switch_pos_x, switch_pos_y, 0]
    img[switch_pos_x, switch_pos_y, 0] = tmp
    assert img.mean()==0.5
    return img

def coffee_cream_sim(n_ims):
    side_length = 100
    top = np.ones([side_length//2, side_length, 1])
    bottom = np.zeros([side_length//2, side_length, 1])
    state = np.concatenate([top, bottom], axis=0)
    for i in range(n_ims):
        for _ in range(50):
            state = switch_rand_pos(state, None, None)
        yield state, i

class ImageStreamer():
    def __init__(self,dset,resize):
        self.dset = dset
        self.resize = resize
        if dset=='im':
            self.dset_dir = 'data/imagenette2/val'
        elif dset=='dtd':
            self.dset_dir = 'data/dtd/suitable'
        elif dset == 'cifar':
            with open('data/cifar-10-batches-py/data_batch_1', 'rb') as f:
                d = pickle.load(f, encoding='bytes')
                imgs = d[b'data']
                imgs = np.transpose(imgs.reshape((-1,3,32,32)),(0,2,3,1))
                labels = d[b'labels']
                self.prepared_dset = list(zip(imgs,labels))
        elif dset == 'mnist':
            imgs, labels = load_mnist()
            self.prepared_dset = list(zip(imgs,labels))
        #elif dset == 'rand':
            #self.prepared_dset = np.random.rand(1000,224,224,3).astype(np.float32).astype(np.float64) # so can fix bitprec at 32
        #elif dset == 'bitrand':
            #self.prepared_dset = (np.random.rand(1000,224,224,3) > 0.5).astype(float)
        elif dset == 'stripes':
            self.line_thicknesses = np.random.permutation(np.arange(70,80))
        elif dset in ('fluidsim', 'fluidsim-blurred'):
            self.im_dir = 'data/fluid-sim'
            if dset=='fluidsim':
                self.im_fns = [x for x in os.listdir(self.im_dir) if 'blurred' not in x]
            else:
                self.im_fns = [x for x in os.listdir(self.im_dir) if 'blurred' in x]
            self.im_fns = natsorted(x for x in self.im_fns if x.endswith('npy'))

    def stream_images(self,num_ims,downsample,given_fname='none',given_class_dir='none',select_randomly=False):
        if self.dset in ('fluidsim', 'fluidsim-blurred'):
            num_ims = min(num_ims, len(self.im_fns))
        if self.dset in ['cifar','mnist','usps']:
            indices = np.random.choice(len(self.prepared_dset),size=num_ims,replace=False)
        elif self.dset == 'dtd':
            n = min(len(listdir('data/dtd/suitable')),num_ims)
            indices = range(n)
        elif self.dset == 'fractal_imgs':
            n = min(len(listdir('fractal_imgs')),num_ims)
            indices = range(n)
        else: indices = range(num_ims)

        for i in indices:
            if self.dset in ['im','dtd']:
                if self.dset=='im':
                    num_classes = len(listdir(self.dset_dir))
                    if select_randomly:
                        class_dir = np.random.choice(listdir(self.dset_dir))
                        fname = np.random.choice(listdir(join(self.dset_dir,class_dir)))
                    else:
                        class_dir = given_class_dir if given_class_dir != 'none' else listdir(self.dset_dir)[i%num_classes]
                        idx_within_class = i//num_classes
                        fname = given_fname if given_fname != 'none' else listdir(join(self.dset_dir,class_dir))[idx_within_class]
                    fpath = join(self.dset_dir,class_dir,fname)
                elif self.dset=='dtd':
                    try:
                        fname = given_fname if given_fname != 'none' else listdir(self.dset_dir)[i]
                    except IndexError:
                        print(f"have run out of images, at image number {i}")
                        sys.exit()
                    fpath = join(self.dset_dir,fname)
                im = load_fpath(fpath,self.resize,downsample)
                im = im/255
                if im.ndim == 2:
                    im = np.resize(im,(*(im.shape),1))
                label = fname
            elif self.dset == 'stripes':
                slope = np.random.rand()+.5
                line_thickness = self.line_thicknesses[i%len(self.line_thicknesses)]
                im = create_simple_img('stripes',slope,line_thickness)
                label = f'stripes-{line_thickness}'
            elif self.dset == 'halves':
                slope = np.random.rand()+.5
                im = create_simple_img('halves',slope,-1)
                label = 'halves'
            elif self.dset =='rand':
                im = np.random.rand(224,224,3).astype(np.float32).astype(np.float64)
                label = 'rand'
            elif self.dset =='bit-rand':
                im = (np.random.rand(224,224,3) > 0.5).astype(float)
                label = 'bit-rand'
            elif self.dset == 'fractal_imgs':
                fname = listdir('fractal_imgs')[i]
                fpath = join('fractal_imgs',fname)
                im = load_fpath(fpath,self.resize,downsample)
                label = 'fract_dim' + fname.split('.')[0][-1]
            elif self.dset in ['cifar']:
                im,label = self.prepared_dset[i]
                im = np.array(Image.fromarray(im).resize((224,224)))/255
            elif self.dset == 'mnist':
                im,label = self.prepared_dset[i]
                if self.resize:
                    im = np.array(Image.fromarray(im).resize((224,224)))
                im = np.tile(np.expand_dims(im,2),(1,1,3))
            elif self.dset in ('fluidsim', 'fluidsim-blurred'):
                im_ = np.load(os.path.join(self.im_dir,self.im_fns[i]))
                print(self.im_fns[i])
                if self.resize:
                    im = np.array(Image.fromarray(im_.astype('float')).resize((224,224)))
                im = np.tile(np.expand_dims(im,2),(1,1,3))
                label = i
            else:
                print("INVALID DSET NAME:", self.dset)
            yield im, label
