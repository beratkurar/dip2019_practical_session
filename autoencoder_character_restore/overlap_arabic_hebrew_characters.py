import h5py
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import numpy as np

# Sources:
# https://github.com/georgeretsi/pytorch-phocnet
# https://github.com/Sarasra/models/tree/master/research/capsules

LIMIT = 3000
DATA_IN_BASE = 'hebrew-letters/'
DATA_IN_TOP = 'arabic-letters/'
DATA_OUT = 'letters-' + str(LIMIT) + '/'
DATA_BASE_HDF5 = 'hebrew-letters.hdf5'
DATA_TOP_HDF5 = 'arabic-letters.hdf5'
DATA_BROKEN_HDF5 = 'broken-letters.hdf5'


def homography(img):
    random_limits = (0.9, 1.1)
    y, x = img.shape[:2]
    fx = float(x)
    fy = float(y)
    src_point = np.float32([[fx / 2, fy / 3, ],
                            [2 * fx / 3, 2 * fy / 3],
                            [fx / 3, 2 * fy / 3]])
    random_shift = (np.random.rand(3, 2) - 0.5) * 2 * (random_limits[1] - random_limits[0]) / 2 + np.mean(random_limits)
    dst_point = src_point * random_shift.astype(np.float32)
    transform = cv2.getAffineTransform(src_point, dst_point)
    border_value = np.median(np.reshape(img, (img.shape[0] * img.shape[1], -1)), axis=0)
    warped_img = cv2.warpAffine(img, transform, dsize=(x, y), borderValue=border_value)
    return warped_img


def create_root_letters(alphabet):
    lettersfile = open(alphabet + '-letters.txt', 'r', encoding='utf-8')
    label = 0
    os.makedirs(alphabet + '-letters/')
    for line in lettersfile:
        line = line.split()
        letter = line[1].rstrip()
        fonts = os.listdir(alphabet + '-fonts/')
        for i in range(len(fonts)):
            font_path = alphabet + '-fonts/' + fonts[i]
            font = ImageFont.truetype(font_path, 60)
            letter_image = Image.new('L', (250, 250), 'black')
            letter_draw = ImageDraw.Draw(letter_image)
            letter_draw.text((90, 90), letter, fill='white', font=font)
            letter_array = np.asarray(letter_image)
            letter_image.close()
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
            morphed_letter = cv2.morphologyEx(letter_array, cv2.MORPH_CLOSE, rect_kernel)
            save_path = alphabet + '-letters/' + str(label) + '-' + str(i) + '.png'
            # cv2.imwrite(save_path, letter_array)
            ret, thresh = cv2.threshold(morphed_letter, 127, 255, 0)
            _, contours, hierarchy = cv2.findContours(thresh, 1, 2)
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)
            roi = letter_array[y - 5:y + h + 5, x - 5:x + w + 5]
            cv2.imwrite(save_path, roi)
        label += 1


def create_hdf5_dataset(alphabet, limit):
    x = []
    y = []
    if alphabet == 'hebrew':
        data_in = DATA_IN_BASE
        data_hdf5 = DATA_BASE_HDF5
    if alphabet == 'arabic':
        data_in = DATA_IN_TOP
        data_hdf5 = DATA_TOP_HDF5
    i = 0
    while i <= limit - 1:
        for letter_name in os.listdir(data_in):
            label = letter_name.split('-')[0]
            letter_image = cv2.imread(data_in + letter_name, 0)
            homog_image = homography(letter_image)
            resized_image = cv2.resize(homog_image, (50, 50))
            # cv2.imwrite(DATA_OUT+label+'-'+str(i)+'.png', resized_image)
            x.append(resized_image)
            y.append(label)
            i += 1
            if i >= limit - 1:
                break
    xa = np.asarray(x)
    ya = np.asarray(y, dtype='uint8')
    f = h5py.File(data_hdf5, 'w')
    f.create_dataset('images', data=xa)
    f.create_dataset('labels', data=ya)
    f.close()


def read_byte_letters(data_file):
    with h5py.File(data_file, 'r') as f:
        images = f['images'][:]
        labels = f['labels'][:]
    return zip(images, labels)


def read_byte_overlaps(data_file):
    with h5py.File(data_file, 'r') as f:
        original_images = f['original_images'][:]
        broken_images = f['broken_images'][:]
        labels = f['labels'][:]
    return broken_images, original_images, labels


def shift_2d(image, shift, max_shift):
    max_shift += 1
    padded_image = np.pad(image, max_shift, 'constant')
    rolled_image = np.roll(padded_image, shift[0], axis=0)
    rolled_image = np.roll(rolled_image, shift[1], axis=1)
    shifted_image = rolled_image[max_shift:-max_shift, max_shift:-max_shift]
    return shifted_image


def create_overlapped_dataset(base_dataset, top_dataset, shift, pad, num_pairs):
    base_list = list(base_dataset)
    top_list = list(top_dataset)
    num_base_images = len(base_list)
    num_top_images = len(top_list)
    random_base_shifts = np.random.randint(-shift, shift + 1, (num_base_images, num_pairs + 1, 2))
    random_top_shifts = np.random.randint(-shift, shift + 1, (num_top_images, num_pairs + 1, 2))
    base_dataset = [(np.pad(image, pad, 'constant'), label) for (image, label) in base_list]
    top_dataset = [(np.pad(image, pad, 'constant'), label) for (image, label) in top_list]
    image_raw_1 = []
    image_raw_2 = []
    merged_raw = []
    label_1 = []
    label_2 = []
    for i, (base_image, base_label) in enumerate(base_dataset):
        base_shifted = shift_2d(base_image, random_base_shifts[i, 0, :], shift).astype(np.uint8)
        choices = np.random.choice(num_top_images, 2 * num_pairs, replace=False)
        chosen_dataset = []
        for choice in choices:
            chosen_dataset.append(top_dataset[choice])
        for j, (top_image, top_label) in enumerate(chosen_dataset[:num_pairs]):
            top_shifted = shift_2d(top_image, random_top_shifts[i, j + 1, :],
                                   shift).astype(np.uint8)
            merged = base_shifted.copy()
            merged[top_shifted > 10] = 0
            #mergedt = np.add(base_shifted, top_shifted, dtype=np.int32)
            #mergedt = np.minimum(mergedt, 255).astype(np.uint8)
            image_raw_1.append(base_shifted)
            image_raw_2.append(top_shifted)
            merged_raw.append(merged)
            label_1.append(base_label)
            label_2.append(top_label)
    image_raw_1a = np.asarray(image_raw_1)
    image_raw_2a = np.asarray(image_raw_2)
    merged_rawa = np.asarray(merged_raw)
    label_1a = np.asarray(label_1, dtype='uint8')
    f = h5py.File(DATA_BROKEN_HDF5, 'w')
    f.create_dataset('broken_images', data=merged_rawa)
    f.create_dataset('original_images', data=image_raw_1a)
    f.create_dataset('labels', data=label_1a)
    f.close()


if not os.path.exists('hebrew-letters'):
    create_root_letters('hebrew')
if not os.path.exists('arabic-letters'):
    create_root_letters('arabic')

if not os.path.exists(DATA_BASE_HDF5):
    create_hdf5_dataset('hebrew', LIMIT)
if not os.path.exists(DATA_TOP_HDF5):
    create_hdf5_dataset('arabic', LIMIT)



if not os.path.exists(DATA_BROKEN_HDF5):
    input_base = read_byte_letters(DATA_BASE_HDF5)
    input_top = read_byte_letters(DATA_TOP_HDF5)
    create_overlapped_dataset(input_base, input_top, 10, 15, 1)



'''''
# Read example
# label is in the name of each letter image. labelnumber-samplenumber.png
overlapped_dataset = read_byte_overlaps(DATA_BROKEN_HDF5)
c = 0
for (bi,oi, l) in overlapped_dataset:
    cv2.imwrite('broken-letters/' + str(l) + '-' + str(c) + '.png', bi)
    cv2.imwrite('original-letters/' + str(l) + '-' + str(c) + '.png', oi)
    c += 1
'''''
