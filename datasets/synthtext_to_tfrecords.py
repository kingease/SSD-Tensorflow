import cv2
import random
import scipy.io as sio
import tensorflow as tf
import numpy as np
import os
import pickle
from tqdm import tqdm
import sys


from datasets.dataset_utils import int64_feature, float_feature, bytes_feature

VOC_LABELS = {
    'none': (0, 'Background'),
    'text': (1, 'text')
}

RANDOM_SEED = 200
SAMPLES_PER_FILES = 500

# convert wordBB to aabb boxes list
def convert_rotbbox_to_aabb(wordBB, image_size):
    if len(wordBB.shape) < 3:
        wordBB = np.expand_dims(wordBB, axis=-1)

    word_len = wordBB.shape[2]
    height, width = image_size
    aabb = np.zeros((word_len, 4), dtype=np.float32)
    for i in xrange(word_len):
        x_min = np.min(wordBB[0, :, i]) / float(width)
        x_max = np.max(wordBB[0, :, i]) / float(width)
        y_min = np.min(wordBB[1, : ,i]) / float(height)
        y_max = np.max(wordBB[1, : ,i]) / float(height)
        aabb[i,:] = np.array([y_min, x_min, y_max, x_max])

    return aabb


def run(dataset_dir, output_dir, split_ratio=0.9, shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    
    def _process_image(directory, idx_of_img):
        """Process a image and annotation file.

        Args:
          directory: the syntext data directory
          idx_of_img: int, the index of images in gt.mat
        Returns:
          image_data: string, JPEG encoding of RGB image.
          shape: the image shape
          bboxes: the bounding box of text
          labels: the label of text always 1, there is only one label
        """
        # Read the image file.
        name = str(gt['imnames'][:,idx_of_img][0][0])
        filename = directory + name
        image_data = tf.gfile.FastGFile(filename, 'r').read()

        shape = image_shapes[name]

        wordBB = gt['wordBB'][:,idx_of_img][0]

        bboxes = convert_rotbbox_to_aabb(wordBB, shape[:2])
        labels = [VOC_LABELS['text'][0]] * len(bboxes)

        return image_data, shape, bboxes, labels
    
    def _convert_to_example(image_data, shape, bboxes, labels):
        """Build an Example proto for an image example.

        Args:
          image_data: string, JPEG encoding of RGB image; Sythtext images are always .jpg
          labels: list of integers, identifier for the ground truth;
          bboxes: list of bounding boxes; each box is a list of integers;
              specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
              to the same label as the image label.
          shape: 3 integers, image shapes in pixels.
        Returns:
          Example proto
        """
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        for b in bboxes:
            assert len(b) == 4
            # pylint: disable=expression-not-assigned
            [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
            # pylint: enable=expression-not-assigned

        image_format = b'JPEG'
        example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': int64_feature(shape[0]),
                'image/width': int64_feature(shape[1]),
                'image/channels': int64_feature(shape[2]),
                'image/shape': int64_feature(shape),
                'image/object/bbox/xmin': float_feature(xmin),
                'image/object/bbox/xmax': float_feature(xmax),
                'image/object/bbox/ymin': float_feature(ymin),
                'image/object/bbox/ymax': float_feature(ymax),
                'image/object/bbox/label': int64_feature(labels),
                'image/format': bytes_feature(image_format),
                'image/encoded': bytes_feature(image_data)}))
        return example
    
    def _add_to_tfrecord(dataset_dir, idx, tfrecord_writer):
        """Loads data from image and annotations files and add them to a TFRecord.

        Args:
          dataset_dir: Dataset directory;
          name: Image name to add to the TFRecord;
          tfrecord_writer: The TFRecord writer to use for writing.
        """
        image_data, shape, bboxes, labels = \
            _process_image(dataset_dir, idx)
        example = _convert_to_example(image_data, shape, bboxes, labels)
        tfrecord_writer.write(example.SerializeToString())
        
    def _get_output_filename(output_dir, split_name, idx):
        return '%s/syntext_%s_%04d.tfrecord' % (output_dir, split_name, idx)

    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    print('read gt.mat file...')
    gt = sio.loadmat(os.path.join(dataset_dir, 'gt.mat'))
    
    print('read image_shape.pkl file...')
    with open(os.path.join(dataset_dir, 'image_shape.pkl')) as f:
        image_shapes = pickle.load(f)
    
    num_of_image = gt['imnames'].shape[1]
    
    assert num_of_image == len(image_shapes)
    
    # assert split_ratio belong to (0, 1)
    num_for_train = int(num_of_image * split_ratio)
    
    # shuffle the indices
    fileidxs = range(0, num_of_image)
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(fileidxs)
    
    # pick up indices for train and test
    idxs_for_train = fileidxs[:num_for_train]
    idxs_for_test = fileidxs[num_for_train:]

    # Process dataset files for trains
    print('convert data for train.')

    i, fidx = 0, 0
    while i < len(idxs_for_train):
        tf_filename = _get_output_filename(output_dir, 'train', fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(idxs_for_train) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(idxs_for_train)))
                sys.stdout.flush()
                _add_to_tfrecord(dataset_dir, idxs_for_train[i], tfrecord_writer)
                i +=1
                j +=1
        
        fidx += 1
    
    # Process dataset files for test
    print('convert data for test.')

    i, fidx = 0, 0
    while i < len(idxs_for_test):
        tf_filename = _get_output_filename(output_dir, 'test', fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(idxs_for_test) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(idxs_for_test)))
                sys.stdout.flush()
                _add_to_tfrecord(dataset_dir, idxs_for_test[i], tfrecord_writer)
                i +=1
                j +=1
        
        fidx += 1
    
    print('\nFinished converting the SynthText dataset!')
