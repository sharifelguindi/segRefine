import fnmatch
import os
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import win32com.client
import h5py
from operator import itemgetter
from PIL import Image, ImageDraw
from imgAug import bbox2_3D
from compress_pickle import load
import numpy as np
from scipy import ndimage
import lookup_tables


def store_arrays_hdf5(data, HDF5_DIR, filename):
    """ Stores python diction of number arrays to HDF5.
        Parameters:
        ---------------
        data    python dictionary, key/value to be stored
    """
    # Create a new HDF5 file
    with h5py.File(os.path.join(HDF5_DIR, filename + '.h5'), 'w') as f:
        for key, value in data.items():
            globals()[key] = f.create_dataset(
                key, np.shape(value), data=value, compression='gzip'
            )
    return


def load_arrays_hdf5(HDF5_DIR, filename):
    data = {}
    with h5py.File(os.path.join(HDF5_DIR, filename), 'r') as f:
        for key in f.keys():
            data[key] = np.array(f[key])

    return data


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def distance_map(mask_gt, spacing_mm, roi_size=None, crop_mask=True):
    # compute the area for all 256 possible surface elements
    # (given a 2x2x2 neighbourhood) according to the spacing_mm
    if roi_size is None:
        roi_size = [64, 64, 64]
    neighbour_code_to_surface_area = np.zeros([256])
    for code in range(256):
        normals = np.array(lookup_tables.neighbour_code_to_normals[code])
        sum_area = 0
        for normal_idx in range(normals.shape[0]):
            # normal vector
            n = np.zeros([3])
            n[0] = normals[normal_idx, 0] * spacing_mm[1] * spacing_mm[2]
            n[1] = normals[normal_idx, 1] * spacing_mm[0] * spacing_mm[2]
            n[2] = normals[normal_idx, 2] * spacing_mm[0] * spacing_mm[1]
            area = np.linalg.norm(n)
            sum_area += area
        neighbour_code_to_surface_area[code] = sum_area

    # compute the bounding box of the masks to trim
    # the volume to the smallest possible processing subvolume
    if crop_mask:
        mask_all = mask_gt
        bbox_min = np.zeros(3, np.int64)
        bbox_max = np.zeros(3, np.int64)

        # max projection to the x0-axis
        proj_0 = np.max(np.max(mask_all, axis=2), axis=1)
        idx_nonzero_0 = np.nonzero(proj_0)[0]
        if len(idx_nonzero_0) == 0:  # pylint: disable=g-explicit-length-test
            return {"distances_gt_to_pred": np.array([]),
                    "distances_pred_to_gt": np.array([]),
                    "surfel_areas_gt": np.array([]),
                    "surfel_areas_pred": np.array([])}

        bbox_min[0] = np.min(idx_nonzero_0)
        bbox_max[0] = np.max(idx_nonzero_0)

        # max projection to the x1-axis
        proj_1 = np.max(np.max(mask_all, axis=2), axis=0)
        idx_nonzero_1 = np.nonzero(proj_1)[0]
        bbox_min[1] = np.min(idx_nonzero_1)
        bbox_max[1] = np.max(idx_nonzero_1)

        # max projection to the x2-axis
        proj_2 = np.max(np.max(mask_all, axis=1), axis=0)
        idx_nonzero_2 = np.nonzero(proj_2)[0]
        bbox_min[2] = np.min(idx_nonzero_2)
        bbox_max[2] = np.max(idx_nonzero_2)

        centroid_contour = [int(np.mean([bbox_min[0], bbox_max[0] + 1])), int(np.mean([bbox_min[1], bbox_max[1] + 1])),
                            int(np.mean([bbox_min[2], bbox_max[2] + 1]))]

        zmin = centroid_contour[0] - int(roi_size[0] / 2)
        zmax = centroid_contour[0] + int(roi_size[0] / 2)
        if zmin < 0:
            zmax = zmax + np.absolute(zmin)
            zmin = 0

        cropmask_gt = mask_gt[zmin:zmax,
                              centroid_contour[1] - int(roi_size[1] / 2):centroid_contour[1] + int(roi_size[1] / 2),
                              centroid_contour[2] - int(roi_size[2] / 2):centroid_contour[2] + int(roi_size[2] / 2)]
    else:
        cropmask_gt = np.copy(mask_gt)
    # crop the processing subvolume.
    # we need to zeropad the cropped region with 1 voxel at the lower,
    # the right and the back side. This is required to obtain the "full"
    # convolution result with the 2x2x2 kernel
    # cropmask_gt = np.zeros((bbox_max - bbox_min) + 2, np.uint8)
    # cropmask_pred = np.zeros((bbox_max - bbox_min) + 2, np.uint8)
    #
    # cropmask_gt[0:-1, 0:-1, 0:-1] = mask_gt[bbox_min[0]:bbox_max[0] + 1,
    #                                 bbox_min[1]:bbox_max[1] + 1,
    #                                 bbox_min[2]:bbox_max[2] + 1]

    # cropmask_pred[0:-1, 0:-1, 0:-1] = mask_pred[bbox_min[0]:bbox_max[0] + 1,
    #                                   bbox_min[1]:bbox_max[1] + 1,
    #                                   bbox_min[2]:bbox_max[2] + 1]

    # compute the neighbour code (local binary pattern) for each voxel
    # the resultsing arrays are spacially shifted by minus half a voxel in each
    # axis.
    # i.e. the points are located at the corners of the original voxels
    kernel = np.array([[[128, 64],
                        [32, 16]],
                       [[8, 4],
                        [2, 1]]])
    neighbour_code_map_gt = ndimage.filters.correlate(
        cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0)
    # neighbour_code_map_pred = ndimage.filters.correlate(
    #     cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0)

    # create masks with the surface voxels
    borders_gt = ((neighbour_code_map_gt != 0) & (neighbour_code_map_gt != 255))
    # borders_pred = ((neighbour_code_map_pred != 0) &
    #                 (neighbour_code_map_pred != 255))

    # compute the distance transform (closest distance of each voxel to the
    # surface voxels)
    if borders_gt.any():
        distmap_gt = ndimage.morphology.distance_transform_edt(
            ~borders_gt, sampling=spacing_mm)

        # make negative distance if inside contour
        distmap_gt[neighbour_code_map_gt == 255] = (-1) * distmap_gt[neighbour_code_map_gt == 255]
    else:
        distmap_gt = []

    return distmap_gt, borders_gt


def compute_surface_distances(mask_gt, mask_pred, spacing_mm):
    """Compute closest distances from all surface points to the other surface.
  Finds all surface elements "surfels" in the ground truth mask `mask_gt` and
  the predicted mask `mask_pred`, computes their area in mm^2 and the distance
  to the closest point on the other surface. It returns two sorted lists of
  distances together with the corresponding surfel areas. If one of the masks
  is empty, the corresponding lists are empty and all distances in the other
  list are `inf`.
  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.
    spacing_mm: 3-element list-like structure. Voxel spacing in x0, x1 and x2
        direction.
  Returns:
    A dict with:
    "distances_gt_to_pred": 1-dim numpy array of type float. The distances in mm
        from all ground truth surface elements to the predicted surface,
        sorted from smallest to largest.
    "distances_pred_to_gt": 1-dim numpy array of type float. The distances in mm
        from all predicted surface elements to the ground truth surface,
        sorted from smallest to largest.
    "surfel_areas_gt": 1-dim numpy array of type float. The area in mm^2 of
        the ground truth surface elements in the same order as
        distances_gt_to_pred
    "surfel_areas_pred": 1-dim numpy array of type float. The area in mm^2 of
        the predicted surface elements in the same order as
        distances_pred_to_gt
  """

    # compute the area for all 256 possible surface elements
    # (given a 2x2x2 neighbourhood) according to the spacing_mm
    neighbour_code_to_surface_area = np.zeros([256])
    for code in range(256):
        normals = np.array(lookup_tables.neighbour_code_to_normals[code])
        sum_area = 0
        for normal_idx in range(normals.shape[0]):
            # normal vector
            n = np.zeros([3])
            n[0] = normals[normal_idx, 0] * spacing_mm[1] * spacing_mm[2]
            n[1] = normals[normal_idx, 1] * spacing_mm[0] * spacing_mm[2]
            n[2] = normals[normal_idx, 2] * spacing_mm[0] * spacing_mm[1]
            area = np.linalg.norm(n)
            sum_area += area
        neighbour_code_to_surface_area[code] = sum_area

    # compute the bounding box of the masks to trim
    # the volume to the smallest possible processing subvolume
    mask_all = mask_gt | mask_pred
    bbox_min = np.zeros(3, np.int64)
    bbox_max = np.zeros(3, np.int64)

    # max projection to the x0-axis
    proj_0 = np.max(np.max(mask_all, axis=2), axis=1)
    idx_nonzero_0 = np.nonzero(proj_0)[0]
    if len(idx_nonzero_0) == 0:  # pylint: disable=g-explicit-length-test
        return {"distances_gt_to_pred": np.array([]),
                "distances_pred_to_gt": np.array([]),
                "surfel_areas_gt": np.array([]),
                "surfel_areas_pred": np.array([])}

    bbox_min[0] = np.min(idx_nonzero_0)
    bbox_max[0] = np.max(idx_nonzero_0)

    # max projection to the x1-axis
    proj_1 = np.max(np.max(mask_all, axis=2), axis=0)
    idx_nonzero_1 = np.nonzero(proj_1)[0]
    bbox_min[1] = np.min(idx_nonzero_1)
    bbox_max[1] = np.max(idx_nonzero_1)

    # max projection to the x2-axis
    proj_2 = np.max(np.max(mask_all, axis=1), axis=0)
    idx_nonzero_2 = np.nonzero(proj_2)[0]
    bbox_min[2] = np.min(idx_nonzero_2)
    bbox_max[2] = np.max(idx_nonzero_2)

    # crop the processing subvolume.
    # we need to zeropad the cropped region with 1 voxel at the lower,
    # the right and the back side. This is required to obtain the "full"
    # convolution result with the 2x2x2 kernel
    cropmask_gt = np.zeros((bbox_max - bbox_min) + 2, np.uint8)
    cropmask_pred = np.zeros((bbox_max - bbox_min) + 2, np.uint8)

    cropmask_gt[0:-1, 0:-1, 0:-1] = mask_gt[bbox_min[0]:bbox_max[0] + 1,
                                            bbox_min[1]:bbox_max[1] + 1,
                                            bbox_min[2]:bbox_max[2] + 1]

    cropmask_pred[0:-1, 0:-1, 0:-1] = mask_pred[bbox_min[0]:bbox_max[0] + 1,
                                                bbox_min[1]:bbox_max[1] + 1,
                                                bbox_min[2]:bbox_max[2] + 1]

    # compute the neighbour code (local binary pattern) for each voxel
    # the resultsing arrays are spacially shifted by minus half a voxel in each
    # axis.
    # i.e. the points are located at the corners of the original voxels
    kernel = np.array([[[128, 64],
                        [32, 16]],
                       [[8, 4],
                        [2, 1]]])
    neighbour_code_map_gt = ndimage.filters.correlate(
        cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0)
    neighbour_code_map_pred = ndimage.filters.correlate(
        cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0)

    # create masks with the surface voxels
    borders_gt = ((neighbour_code_map_gt != 0) & (neighbour_code_map_gt != 255))
    borders_pred = ((neighbour_code_map_pred != 0) &
                    (neighbour_code_map_pred != 255))

    # compute the distance transform (closest distance of each voxel to the
    # surface voxels)
    if borders_gt.any():
        distmap_gt = ndimage.morphology.distance_transform_edt(
            ~borders_gt, sampling=spacing_mm)
    else:
        distmap_gt = np.Inf * np.ones(borders_gt.shape)

    if borders_pred.any():
        distmap_pred = ndimage.morphology.distance_transform_edt(
            ~borders_pred, sampling=spacing_mm)
    else:
        distmap_pred = np.Inf * np.ones(borders_pred.shape)

    # compute the area of each surface element
    surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
    surface_area_map_pred = neighbour_code_to_surface_area[
        neighbour_code_map_pred]

    # create a list of all surface elements with distance and area
    distances_gt_to_pred = distmap_pred[borders_gt]
    distances_pred_to_gt = distmap_gt[borders_pred]
    surfel_areas_gt = surface_area_map_gt[borders_gt]
    surfel_areas_pred = surface_area_map_pred[borders_pred]

    # sort them by distance
    if distances_gt_to_pred.shape != (0,):
        sorted_surfels_gt = np.array(
            sorted(zip(distances_gt_to_pred, surfel_areas_gt)))
        distances_gt_to_pred = sorted_surfels_gt[:, 0]
        surfel_areas_gt = sorted_surfels_gt[:, 1]

    if distances_pred_to_gt.shape != (0,):
        sorted_surfels_pred = np.array(
            sorted(zip(distances_pred_to_gt, surfel_areas_pred)))
        distances_pred_to_gt = sorted_surfels_pred[:, 0]
        surfel_areas_pred = sorted_surfels_pred[:, 1]

    return {"distances_gt_to_pred": distances_gt_to_pred,
            "distances_pred_to_gt": distances_pred_to_gt,
            "surfel_areas_gt": surfel_areas_gt,
            "surfel_areas_pred": surfel_areas_pred}


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def loadContour(contourFile_ref, scanFile, HDF5_DIR, returnMask=True):
    if returnMask:
        fillValue = 1
    else:
        fillValue = 0

    pad = [0, 0, 0, 0, 0, 0]

    scanData = load_arrays_hdf5(HDF5_DIR, scanFile)
    coordTransform = scanData['coordinateSystemTransform']
    imageSize = scanData['imageSize']
    sliceLocations = scanData['sliceLocations']
    ref = load(open(os.path.join(HDF5_DIR, contourFile_ref), 'rb'), compression='gzip')
    max_z = max(ref, key=itemgetter(0))[0]
    min_z = min(ref, key=itemgetter(0))[0]

    pxSpacing = [np.round(np.mean(coordTransform[0, 0, :]), 2),
                 np.round(np.mean(coordTransform[1, 1, :]), 2),
                 np.round(np.mean(np.diff(coordTransform[2, 3, :])), 2)]

    mask_ref = np.zeros(imageSize)

    for z in np.arange(min_z, max_z + pxSpacing[2], pxSpacing[2]):
        rndDig = 3
        z = np.round(z, rndDig)
        if z <= sliceLocations[-1]:
            slicePts_ref = [item for item in ref if item[0] == z]
            zLoc = np.where(np.round(coordTransform[2, 3, :], rndDig) == z)

            # Verify z location through continual rounding for robustness
            if zLoc[0][0] >= 0:
                pxTransform = coordTransform[:, :, zLoc[0][0]]
            else:
                rndDig = 2
                z = np.round(z, rndDig)
                zLoc = np.where(np.round(coordTransform[2, 3, :], rndDig) == z)
                if zLoc[0][0] >= 0:
                    pxTransform = coordTransform[:, :, zLoc[0][0]]
                else:
                    rndDig = 1
                    z = np.round(z, rndDig)
                    zLoc = np.where(np.round(coordTransform[2, 3, :], rndDig) == z)
                    pxTransform = coordTransform[:, :, zLoc[0][0]]

            # pxTransform = coordTransform[:, :, np.where(coordTransform[2, 3, :] == z)][:, :, 0, 0]
            pxTransform_inv = np.linalg.pinv(pxTransform)

            if slicePts_ref:
                n_cur_pts = int(len(slicePts_ref[0][1]) / 3)
                cur_contour = slicePts_ref[0][1]
                cur_contour_2_d = []
                for i in range(0, n_cur_pts):
                    coord = [float(cur_contour[i * 3]), float(cur_contour[i * 3 + 1]), float(cur_contour[i * 3 + 2]), 1]
                    pxCoord = np.matmul(pxTransform_inv, coord)
                    cur_contour_2_d.append((np.round(pxCoord[0]), np.round(pxCoord[1])))
                img = Image.new('L', (imageSize[1], imageSize[2]), 0)
                ImageDraw.Draw(img).polygon(cur_contour_2_d, outline=1, fill=fillValue)
                mask_ref[zLoc, :, :, 0] = np.array(img)

    rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(mask_ref, pad)
    bbox = [rmin, rmax, cmin, cmax, zmin, zmax]

    return mask_ref, bbox, pxSpacing, scanData


def loadContourQAImg(contourFile_ref, contourFile_test, scanFile, HDF5_DIR, returnMask=True):
    if returnMask:
        fillValue = 1
    else:
        fillValue = 0

    pad = [2, 2, 10, 10, 10, 10]

    scanData = load_arrays_hdf5(HDF5_DIR, scanFile)
    coordTransform = scanData['coordinateSystemTransform']
    imageSize = scanData['imageSize']
    sliceLocations = scanData['sliceLocations']
    scan = scanData['imageMatrix'] + scanData['pixelConversion'][0]
    ref = load(open(os.path.join(HDF5_DIR, contourFile_ref), 'rb'), compression='gzip')
    test = load(open(os.path.join(HDF5_DIR, contourFile_test), 'rb'), compression='gzip')

    max_z = max(max(ref, key=itemgetter(0))[0], max(test, key=itemgetter(0))[0])
    min_z = min(min(ref, key=itemgetter(0))[0], min(test, key=itemgetter(0))[0])

    pxSpacing = [np.round(np.mean(coordTransform[0, 0, :]), 2),
                 np.round(np.mean(coordTransform[1, 1, :]), 2),
                 np.round(np.mean(np.diff(coordTransform[2, 3, :])), 2)]

    mask_ref = np.zeros(imageSize)
    mask_test = np.zeros(imageSize)

    for z in np.arange(min_z, max_z + pxSpacing[2], pxSpacing[2]):
        z = np.round(z, 2)
        slicePts_ref = [item for item in ref if item[0] == z]
        slicePts_test = [item for item in test if item[0] == z]
        pxTransform = coordTransform[:, :, np.where(coordTransform[2, 3, :] == z)][:, :, 0, 0]
        pxTransform_inv = np.linalg.pinv(pxTransform)

        if slicePts_ref:
            n_cur_pts = int(len(slicePts_ref[0][1]) / 3)
            cur_contour = slicePts_ref[0][1]
            cur_contour_2_d = []
            for i in range(0, n_cur_pts):
                coord = [float(cur_contour[i * 3]), float(cur_contour[i * 3 + 1]), float(cur_contour[i * 3 + 2]), 1]
                pxCoord = np.matmul(pxTransform_inv, coord)
                cur_contour_2_d.append((np.round(pxCoord[0]), np.round(pxCoord[1])))
            img = Image.new('L', (512, 512), 0)
            ImageDraw.Draw(img).polygon(cur_contour_2_d, outline=1, fill=fillValue)
            mask_ref[np.where(sliceLocations == z), :, :, 0] = np.array(img)

        if slicePts_test:
            n_cur_pts = int(len(slicePts_test[0][1]) / 3)
            cur_contour = slicePts_test[0][1]
            cur_contour_2_d = []
            for i in range(0, n_cur_pts):
                coord = [float(cur_contour[i * 3]), float(cur_contour[i * 3 + 1]), float(cur_contour[i * 3 + 2]), 1]
                pxCoord = np.matmul(pxTransform_inv, coord)
                cur_contour_2_d.append((np.round(pxCoord[0]), np.round(pxCoord[1])))
            img = Image.new('L', (512, 512), 0)
            ImageDraw.Draw(img).polygon(cur_contour_2_d, outline=1, fill=fillValue)
            mask_test[np.where(sliceLocations == z), :, :, 0] = np.array(img)

    rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(mask_ref + mask_test, pad)
    bbox = [rmin, rmax, cmin, cmax, zmin, zmax]

    return mask_ref, mask_test, scan, bbox


def word_doc_toDocx(filepath):
    word = win32com.client.Dispatch("Word.application")
    docxFilepath = '{0}{1}'.format(filepath, 'x')
    try:
        wordDoc = word.Documents.Open(filepath, False, False, False)
        wordDoc.SaveAs2(docxFilepath, FileFormat=16)
        wordDoc.Close()
    except Exception:
        print('Failed to Convert: {0}'.format(filepath))

    return docxFilepath


def get_single_metric_by_group(df, structList, metricName, rowIndex):
    dff = pd.DataFrame(columns=structList)
    metricDict = {}
    for struct in structList:
        df_by_struct = df[struct + '_' + metricName]
        if df_by_struct.iloc[rowIndex] != 'nan' and isinstance(df_by_struct.iloc[rowIndex], str):
            formatted_str = df_by_struct.iloc[rowIndex].strip('[').strip(']').replace('   ', ' ').replace('  ',
                                                                                                          ' ').split(
                sep=' ')
            if len(formatted_str) > 3:
                formatted_str = formatted_str[0:3]
            g_cen_struct = [float(i) for i in formatted_str]
            metricDict[struct] = g_cen_struct
        elif is_number(df_by_struct.iloc[rowIndex]):
            metricDict[struct] = df_by_struct.iloc[rowIndex]
        else:
            metricDict[struct] = None
    return dff.append(metricDict, ignore_index=True), metricDict


def addMetricData(dbFileName):
    df = pd.read_excel(dbFileName)
    filtered_col = [col for col in df if col.split(',')[0].endswith('DSC')]
    metricFile = os.path.join(df['basePath'].iloc[0], df['dataFilename'].iloc[0]).replace('data', 'metric').replace(
        '.mat', '.csv')
    metricDF = pd.read_csv(metricFile)
    columns_to_add = []
    for col in filtered_col:
        for metric in metricDF.columns:
            if metric != 'Organ':
                columns_to_add.append(col.split(',')[0].replace('_DSC', '') + '_' + metric)

    for col in columns_to_add:
        df[col] = ""
    i = 0
    for _ in df['MRN']:
        if isinstance(df['basePath'][i], str):
            metricFile = os.path.join(df['basePath'].iloc[i], df['dataFilename'].iloc[i]).replace('data',
                                                                                                  'metric').replace(
                '.mat', '.csv')
            metricDF = pd.read_csv(metricFile)
            j = 0
            for organ in metricDF['Organ']:
                k = 0
                for metric in metricDF.columns:
                    if metric != 'Organ':
                        columnName = organ + "_" + metric
                        df[columnName].iloc[i] = metricDF.iloc[j, k]
                    k = k + 1
                j = j + 1
        i = i + 1

    return df


# noinspection PyTypeChecker
def plotTrainingProgress(modelDir, metric):
    # First, gather all csv in dir and sort by ascending checkpoint value
    files = find('*.csv', modelDir)
    checkPoints = {}
    for file in files:
        checkPoint = int(file.split(os.sep)[-1].replace('.csv', ''))
        checkPoints[checkPoint] = file

    # Read one csv, get structures calculated, create empty DF
    df = pd.read_csv(files[0]).drop(['Unnamed: 0', 'Name'], axis=1)
    columns = ['checkPoint']
    for col in df.columns:
        columns.append(col)
    data = pd.DataFrame(columns=columns)

    for key in sorted(checkPoints.keys()):
        df = pd.read_csv(checkPoints[key])
        dscScores = df[df['Name'].str.contains(metric)].drop(['Unnamed: 0', 'Name'], axis=1).describe()
        dscScores['checkPoint'] = int(key)
        data = data.append(dscScores.iloc[1, :])

    ax = plt.gca()
    for col in columns:
        if col != 'checkPoint':
            data.plot(kind='line', x='checkPoint', y=col, ax=ax)
    plt.legend(fontsize='xx-small')
    plt.show()
    return data


def main():
    print('This file consists of helper functions for DICOM RTSTRUCT QA, visualization and DL Applications')


if __name__ == '__main__':
    main()
