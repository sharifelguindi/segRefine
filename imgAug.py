from __future__ import print_function
import os
import cv2
import PIL
import SimpleITK as sitk
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from skimage.transform import rescale, resize


# img is 3D boolean array, pad is vector: [-r ,+r, -c, +c , -z, +z]
def bbox2_3D(img, pad):

    imgShape = np.shape(img)
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    rmin = rmin - pad[0]
    if rmin < 0:
        rmin = 0

    rmax = rmax + pad[1]
    if rmax > imgShape[0]:
        rmax = imgShape[0]

    cmin = cmin - pad[2]
    if cmin < 0:
        cmin = 0

    cmax = cmax + pad[3]
    if cmax > imgShape[1]:
        cmax = imgShape[1]

    zmin = zmin - pad[4]
    if zmin < 0:
        zmin = 0

    zmax = zmax + pad[5]
    if zmax > imgShape[2]:
        zmax = imgShape[2]

    return rmin, rmax, cmin, cmax, zmin, zmax


def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowImage = image.copy()
    windowImage[windowImage < img_min] = img_min
    windowImage[windowImage > img_max] = img_max

    return windowImage


def normalize_equalize_smooth_CT(arr, clahe, WL):

    length, width, height = np.shape(arr)
    norm_arr = window_image(arr, WL[0], WL[1])
    norm_eq = np.zeros((length, 3, width, height), dtype='float')
    if (clahe is not None) and (clahe != 1):
        for ii in range(0, length):
            if ii == 0:
                img_0 = clahe.apply(norm_arr[ii+0, :, :].astype('uint16'))
                img_1 = clahe.apply(norm_arr[ii+0, :, :].astype('uint16'))
                img_2 = clahe.apply(norm_arr[ii+1, :, :].astype('uint16'))
            elif ii == length-1:
                img_0 = clahe.apply(norm_arr[ii-1, :, :].astype('uint16'))
                img_1 = clahe.apply(norm_arr[ii, :, :].astype('uint16'))
                img_2 = clahe.apply(norm_arr[ii, :, :].astype('uint16'))
            else:
                img_0 = clahe.apply(norm_arr[ii-1, :, :].astype('uint16'))
                img_1 = clahe.apply(norm_arr[ii+0, :, :].astype('uint16'))
                img_2 = clahe.apply(norm_arr[ii+1, :, :].astype('uint16'))
            norm_eq[ii, 0, :, :] = smooth_image(img_0).astype('uint16')
            norm_eq[ii, 1, :, :] = 255 - smooth_image(img_1).astype('uint16')
            norm_eq[ii, 2, :, :] = smooth_image(img_2).astype('uint16')
    elif clahe == 1:
        for ii in range(0, length):
            if ii == 0:
                img_0 = norm_arr[ii+0, :, :]
                img_1 = norm_arr[ii+0, :, :]
                img_2 = norm_arr[ii+0, :, :]
            elif ii == length-1:
                img_0 = norm_arr[ii-0, :, :]
                img_1 = norm_arr[ii, :, :]
                img_2 = norm_arr[ii, :, :]
            else:
                img_0 = norm_arr[ii-0, :, :]
                img_1 = norm_arr[ii+0, :, :]
                img_2 = norm_arr[ii+0, :, :]

            norm_eq[ii, 0, :, :] = img_0
            norm_eq[ii, 1, :, :] = img_1
            norm_eq[ii, 2, :, :] = img_2
    else:
        for ii in range(0, length):
            if ii == 0:
                img_0 = norm_arr[ii+0, :, :].astype('uint16')
                img_1 = norm_arr[ii+0, :, :].astype('uint16')
                img_2 = norm_arr[ii+1, :, :].astype('uint16')
            elif ii == length-1:
                img_0 = norm_arr[ii-1, :, :].astype('uint16')
                img_1 = norm_arr[ii, :, :].astype('uint16')
                img_2 = norm_arr[ii, :, :].astype('uint16')
            else:
                img_0 = norm_arr[ii-1, :, :].astype('uint16')
                img_1 = norm_arr[ii+0, :, :].astype('uint16')
                img_2 = norm_arr[ii+1, :, :].astype('uint16')

            norm_eq[ii, 0, :, :] = smooth_image(img_0).astype('uint16')
            norm_eq[ii, 1, :, :] = 255 - smooth_image(img_1).astype('uint16')
            norm_eq[ii, 2, :, :] = smooth_image(img_2).astype('uint16')

    norm_arr_ds = np.zeros(np.shape(norm_eq), dtype='uint8')
    norm_arr_ds = cv2.normalize(norm_eq, norm_arr_ds, 0, 255, cv2.NORM_MINMAX)
    return norm_arr_ds.astype('uint8')


def normalize_equalize_smooth_MR(arr, clahe, pct_remove):

    length, width, height = np.shape(arr)
    norm_arr = np.zeros(np.shape(arr), dtype='uint16')
    norm_arr = cv2.normalize(arr, norm_arr, 0, 65535, cv2.NORM_MINMAX)
    hist, bin_edges = np.histogram(norm_arr[:], 255)
    cum_sum = np.cumsum(hist.astype('float') / np.sum(hist).astype('float'))
    clip_value = bin_edges[np.min(np.where(cum_sum > pct_remove))]
    norm_arr = np.clip(norm_arr, 0, clip_value)
    norm_eq = np.zeros((length, 3, width, height), dtype='uint16')
    if clahe is not None:
        for ii in range(0, length):
            if ii == 0:
                img_0 = clahe.apply(norm_arr[ii+0, :, :].astype('uint16'))
                img_1 = clahe.apply(norm_arr[ii+0, :, :].astype('uint16'))
                img_2 = clahe.apply(norm_arr[ii+1, :, :].astype('uint16'))
            elif ii == length-1:
                img_0 = clahe.apply(norm_arr[ii-1, :, :].astype('uint16'))
                img_1 = clahe.apply(norm_arr[ii, :, :].astype('uint16'))
                img_2 = clahe.apply(norm_arr[ii, :, :].astype('uint16'))
            else:
                img_0 = clahe.apply(norm_arr[ii-1, :, :].astype('uint16'))
                img_1 = clahe.apply(norm_arr[ii+0, :, :].astype('uint16'))
                img_2 = clahe.apply(norm_arr[ii+1, :, :].astype('uint16'))

            norm_eq[ii, 0, :, :] = smooth_image(img_0).astype('uint16')
            norm_eq[ii, 1, :, :] = 255 - smooth_image(img_1).astype('uint16')
            norm_eq[ii, 2, :, :] = smooth_image(img_2).astype('uint16')
    else:
        for ii in range(0, length):
            if ii == 0:
                img_0 = norm_arr[ii+0, :, :].astype('uint16')
                img_1 = norm_arr[ii+0, :, :].astype('uint16')
                img_2 = norm_arr[ii+1, :, :].astype('uint16')
            elif ii == length-1:
                img_0 = norm_arr[ii-1, :, :].astype('uint16')
                img_1 = norm_arr[ii, :, :].astype('uint16')
                img_2 = norm_arr[ii, :, :].astype('uint16')
            else:
                img_0 = norm_arr[ii-1, :, :].astype('uint16')
                img_1 = norm_arr[ii+0, :, :].astype('uint16')
                img_2 = norm_arr[ii+1, :, :].astype('uint16')

            norm_eq[ii, 0, :, :] = smooth_image(img_0).astype('uint16')
            norm_eq[ii, 1, :, :] = 255 - smooth_image(img_1).astype('uint16')
            norm_eq[ii, 2, :, :] = smooth_image(img_2).astype('uint16')

    norm_arr_ds = np.zeros(np.shape(norm_eq), dtype='uint8')
    norm_arr_ds = cv2.normalize(norm_eq, norm_arr_ds, 0, 255, cv2.NORM_MINMAX)
    return norm_arr_ds.astype('uint8')


def smooth_image(arr, t_step=0.125, n_iter=3):
    img = sitk.GetImageFromArray(arr)
    img = sitk.CurvatureFlow(image1=img,
                             timeStep=t_step,
                             numberOfIterations=n_iter)
    arr_smoothed = sitk.GetArrayFromImage(img)
    return arr_smoothed


def data_export_MR_3D(scan, mask, save_path, p_num, struct_name, num_classes):
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 7)),
                               # blur image using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(3, 11)),
                               # blur image using local medians with kernel sizes between 2 and 7
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                           # search either for all edges or for directed edges,
                           # blend the result with the original image using a blobby mask
                           iaa.SimplexNoiseAlpha(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0.5, 1.0)),
                               iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           ])),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                           # add gaussian noise to images
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                           ]),
                           iaa.Invert(0.05, per_channel=True),  # invert color channels
                           iaa.Add((-10, 10), per_channel=0.5),
                           # change brightness of images (by -10 to 10 of original value)
                           iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           iaa.OneOf([
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               iaa.FrequencyNoiseAlpha(
                                   exponent=(-4, 0),
                                   first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                   second=iaa.LinearContrast((0.5, 2.0))
                               )
                           ]),
                           iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           iaa.Grayscale(alpha=(0.0, 1.0)),
                           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                           # move pixels locally around (with random strengths)
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           # sometimes move parts of the image around
                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )

    # Parameters for dataset
    length, height, width = np.shape(scan)
    train_size = 480
    max_class = np.max(mask)

    # Create folders to store images/masks
    save_path = os.path.join(save_path, struct_name, 'processed')
    if not os.path.exists(os.path.join(save_path,'PNGImages')):
        os.makedirs(os.path.join(save_path,'PNGImages'))
    if not os.path.exists(os.path.join(save_path, 'SegmentationClass')):
        os.makedirs(os.path.join(save_path, 'SegmentationClass'))
    if not os.path.exists(os.path.join(save_path, 'SegmentationVis')):
        os.makedirs(os.path.join(save_path, 'SegmentationVis'))

    # Verify size of scan data and mask data equivalent
    if scan.shape == mask.shape:

        clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(int(scan.shape[1] / 16), int(scan.shape[2] / 16)))
        scan_norm = normalize_equalize_smooth_MR(scan, clahe, 0.999)

        images = scan_norm.transpose(0, 3, 2, 1)

        if height != train_size:
            images = (resize(images, (length, train_size, train_size, 3), anti_aliasing=True, order=2)*255.0).astype('uint8')

        segmaps = np.expand_dims(mask, axis=3).astype(np.int32)

        if height != train_size:
            segmaps = resize(mask, (length, train_size, train_size, 1), anti_aliasing=True, order=0).astype(np.int32)

        segmaps = segmaps.transpose(0, 2, 1, 3).astype(np.int32)

        addAug = True
        for numAug in range(0, 10):
            images_aug, segmaps_aug = seq(images=images, segmentation_maps=segmaps.astype(np.int32))

            if numAug == 0:
                for i in range(0, length):
                    img = images[i, :, :, :]
                    seg = segmaps[i, :, :, 0]
                    if np.max(seg) > 0:
                        img_name = os.path.join('PNGImages', 'ax' + str(p_num) + '_' + str(i))
                        gt_name = os.path.join('SegmentationClass', 'ax' + str(p_num) + '_' + str(i))
                        vis_name = os.path.join('SegmentationVis', 'ax' + str(p_num) + '_' + str(i))
                        PIL.Image.fromarray(img).save(os.path.join(save_path, img_name + '.png'))
                        PIL.Image.fromarray(seg.astype('uint8')).save(os.path.join(save_path, gt_name + '.png'))
                        PIL.Image.fromarray((seg*(255/num_classes)).astype('uint8')).save(os.path.join(save_path, vis_name + '.png'))

            if addAug:
                for i in range(0, length):
                    img = images_aug[i, :, :, :]
                    seg = segmaps_aug[i, :, :, 0]
                    if np.max(seg) > 0:
                        img_name = os.path.join('PNGImages', 'ax' + str(p_num) + '_' + str(i) + '_' + str(numAug))
                        gt_name = os.path.join('SegmentationClass', 'ax' + str(p_num) + '_' + str(i) + '_' + str(numAug))
                        vis_name = os.path.join('SegmentationVis', 'ax' + str(p_num) + '_' + str(i) + '_' + str(numAug))
                        PIL.Image.fromarray(img).save(os.path.join(save_path, img_name + '.png'))
                        PIL.Image.fromarray(seg.astype('uint8')).save(os.path.join(save_path, gt_name + '.png'))
                        PIL.Image.fromarray((seg*(255/num_classes)).astype('uint8')).save(os.path.join(save_path, vis_name + '.png'))


