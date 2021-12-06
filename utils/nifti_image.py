'''
Author: Samuel Remedios

Extends ImageDataGenerator and flow_from_directory() to handle the NIfTI file format
by means of the NiBabel package.

Extension of the keras-preprocessing image utilities available at:
https://github.com/keras-team/kears-preprocessing/blob/master/keras_preprocessing/image.py
'''

import keras.backend as K
from keras.preprocessing.image import *
import nibabel as nib
import os
import numpy as np
import threading
import warnings
import multiprocessing.pool
from functools import partial

from PIL import Image as pil_image 

from .display import show_image
from .pad import pad_image

backend = K

class NIfTIDirectoryIterator(Iterator):
    def __init__(self, directory, image_data_generator,
                 augmentations=None,
                 target_size=(240, 240, 155), num_channels=1,
                 num_patches=1,
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 axial_slice=None,
                 follow_links=False, split=None):
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.augmentations = augmentations
        self.target_size = tuple(target_size)
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.image_shape = self.target_size + (num_channels,)
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.axial_slice = axial_slice

        white_list_formats = {'nii', 'nii.gz'}

        # Counter number of samples and classes
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()

        function_partial = partial(_count_valid_files_in_directory,
                                   white_list_formats=white_list_formats,
                                   follow_links=follow_links,
                                   split=None)
        self.samples = sum(pool.map(function_partial,
                                    (os.path.join(directory, subdir)
                                     for subdir in classes)))
        print('Found %d images belonging to %d classes.' %
              (self.samples, self.num_classes))

        # Build an index of the images in the different class subfolders
        results = []
        self.filenames = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(
                pool.apply_async(_list_valid_filenames_in_directory,
                                 (dirpath, white_list_formats, split,
                                  self.class_indices, follow_links)))
        for res in results:
            classes, filenames = res.get()
            self.classes[i:i + len(classes)] = classes
            self.filenames += filenames
            i += len(classes)

        pool.close()
        pool.join()

        super(NIfTIDirectoryIterator, self).__init__(self.samples,
                                                     batch_size,
                                                     shuffle,
                                                     seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array) * self.num_patches,) + self.image_shape,
            dtype=backend.floatx())
        # build batch of image data
       
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            #print(fname)
            img = nib.load(os.path.join(self.directory, fname))
            # TODO: pad/crop image to target_size here
            x = img.get_data()
            x = pad_image(x, self.target_size)
            #x = pad_image(x, (256,256,256))
            # TODO: extensible image augmentation applied here
            if self.augmentations:
                for aug, params in self.augmentations.items():
                    x = aug(x, **params)

            # TODO: figure out how to handle multi-chan data; for now 3D volumes with 1 chan
            
            x = np.reshape(x, x.shape + (self.num_channels,))
            batch_x[i] = x
       


        if self.save_to_dir:
            for i, j in enumerate(index_array):

                # if 3D training images, save a single slice to disk
                if len(batch_x.shape) == 5:
                    if self.axial_slice:
                        img = batch_x[i,:,:,self.axial_slice,0].T
                    else:
                        middle_slice = batch_x.shape[3]//2
                        img = batch_x[i,:,:,middle_slice,0].T
                    fname = '{prefix}_{index}_{hash}.{format}'.format(
                            prefix=self.save_prefix,
                            index=j,
                            hash=np.random.randint(1e7),
                            format=self.save_format)
                    fname = os.path.join(self.save_to_dir, fname)
                    #imsave(fname, img)
                # if 2D training images, save that 2D image to disk
                elif len(batch_x.shape) == 4: 
                    img = batch_x[i,:,:,0].T
                    fname = '{prefix}_{index}_{hash}.{format}'.format(
                            prefix=self.save_prefix,
                            index=j,
                            hash=np.random.randint(1e7),
                            format=self.save_format)
                    fname = os.path.join(self.save_to_dir, fname)
                    #imsave(fname, img)

        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(backend.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=backend.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)


class NIfTIImageDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory,
                            augmentations=None,
                            target_size=(240, 240, 155), num_channels=1,
                            num_patches=1,
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='png',
                            axial_slice=None,
                            follow_links=False):
        return NIfTIDirectoryIterator(directory, self,
                                      augmentations=augmentations,
                                      target_size=target_size, num_channels=num_channels,
                                      num_patches=num_patches,
                                      classes=classes, class_mode=class_mode,
                                      batch_size=batch_size, shuffle=shuffle, seed=seed,
                                      save_to_dir=save_to_dir, save_prefix=save_prefix, 
                                      save_format=save_format,
                                      axial_slice=axial_slice,
                                      follow_links=follow_links)

    def apply_transform(self, x, transform_parameters):
        pass

    def get_random_transform(self, img_shape, seed=None):
        pass

    def standardize(self, x):
        pass



class NIfTINumpyArrayIterator(NumpyArrayIterator):
    # TODO: change some stuff to work with nibabel
    def __init__(self):
        raise NotImplementedError


def _iter_valid_files(directory, white_list_formats, follow_links):
    """Iterates on files with extension in `white_list_formats` contained in `directory`.

   # Arguments
    directory: Absolute path to the directory
    containing files to be counted
    white_list_formats: Set of strings containing allowed extensions for
    the files to be counted.
    follow_links: Boolean.

   # Yields
    Tuple of (root, filename) with extension in `white_list_formats`.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda x: x[0])

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            for extension in white_list_formats:
                if fname.lower().endswith('.tiff'):
                    warnings.warn('Using \'.tiff\' files with multiple bands '
                                  'will cause distortion. '
                                  'Please verify your output.')
                if fname.lower().endswith('.' + extension):
                    yield root, fname

def _count_valid_files_in_directory(directory,
                                    white_list_formats,
                                    split,
                                    follow_links):
    ''' has to be implemented again just because of how partial works'''
    num_files = len(list(
        _iter_valid_files(directory, white_list_formats, follow_links)))
    if split:
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
    else:
        start, stop = 0, num_files
    return stop - start


def _list_valid_filenames_in_directory(directory, white_list_formats, split,
                                       class_indices, follow_links):
    """Lists paths of files in `subdir` with extensions in `white_list_formats`.

    # Arguments
    directory: absolute path to a directory containing the files to list.
    The directory name is used as class label
    and must be a key of `class_indices`.
    white_list_formats: set of strings containing allowed extensions for
    the files to be counted.
    split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
    account a certain fraction of files in each directory.
    E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
    of images in each directory.
    class_indices: dictionary mapping a class name to its index.
    follow_links: boolean.

   # Returns
   classes: a list of class indices
   filenames: the path of valid files in `directory`, relative from
   `directory`'s parent (e.g., if `directory` is "dataset/class1",
   the filenames will be
    `["class1/file1.jpg", "class1/file2.jpg", ...]`).
    """
    dirname = os.path.basename(directory)
    if split:
        num_files = len(list(
            _iter_valid_files(directory, white_list_formats, follow_links)))
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = list(
            _iter_valid_files(
                directory, white_list_formats, follow_links))[start: stop]
    else:
        valid_files = _iter_valid_files(
            directory, white_list_formats, follow_links)

    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)

    return classes, filenames

