import fnmatch
import os
import pandas as pd
from shapely.geometry import *
import matplotlib.pyplot as plt
import numpy as np
import os.path
import win32com.client
import pydicom
import h5py
import pickle
from operator import itemgetter
from PIL import Image, ImageDraw
from imgAug import bbox2_3D
from compress_pickle import dump, load


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


def getKey(item):
    return item[0]


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


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

    pad = [2, 2, 10, 10, 10, 10]

    scanData = load_arrays_hdf5(HDF5_DIR, scanFile)
    coordTransform = scanData['coordinateSystemTransform']
    imageSize = scanData['imageSize']
    sliceLocations = scanData['sliceLocations']
    scan = scanData['imageMatrix'] + scanData['pixelConversion'][0]
    ref = load(open(os.path.join(HDF5_DIR, contourFile_ref), 'rb'), compression='gzip')
    max_z = max(ref, key=itemgetter(0))[0]
    min_z = min(ref, key=itemgetter(0))[0]

    pxSpacing = [np.round(np.mean(coordTransform[0, 0, :]), 2),
                 np.round(np.mean(coordTransform[1, 1, :]), 2),
                 np.round(np.mean(np.diff(coordTransform[2, 3, :])), 2)]

    mask_ref = np.zeros(imageSize)

    for z in np.arange(min_z, max_z + pxSpacing[2], pxSpacing[2]):
        z = np.round(z, 2)
        slicePts_ref = [item for item in ref if item[0] == z]
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

    rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(mask_ref, pad)
    bbox = [rmin, rmax, cmin, cmax, zmin, zmax]

    return mask_ref, scan, bbox


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

    rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(mask_ref+mask_test, pad)
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
            formatted_str = df_by_struct.iloc[rowIndex].strip('[').strip(']').replace('   ', ' ').replace('  ', ' ').split(sep=' ')
            if len(formatted_str) > 3:
                formatted_str = formatted_str[0:3]
            g_cen_struct = [float(i) for i in formatted_str]
            metricDict[struct] = g_cen_struct
        elif is_number(df_by_struct.iloc[rowIndex]):
            metricDict[struct] = df_by_struct.iloc[rowIndex]
        else:
            metricDict[struct] = None
    return dff.append(metricDict, ignore_index=True), metricDict


def plot_polygons_and_linestrings(structure_to_plot, color_for_plot='#00000'):
    if isinstance(structure_to_plot, MultiLineString):
        for bit_to_plot in structure_to_plot:
            x, y = bit_to_plot.xy
            plt.plot(x, y, color=color_for_plot)
    elif isinstance(structure_to_plot, MultiPolygon):
        for bit_to_plot in structure_to_plot:
            x, y = bit_to_plot.boundary.xy
            plt.plot(x, y, color=color_for_plot)
    elif isinstance(structure_to_plot, Polygon):
        x, y = structure_to_plot.boundary.xy
        plt.plot(x, y, color=color_for_plot)
    elif isinstance(structure_to_plot, LineString):
        x, y = structure_to_plot.xy
        plt.plot(x, y, color=color_for_plot)
    else:
        print('Unable to plot structure type: ', type(structure_to_plot))


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
    for pID in df['MRN']:
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


def store_rtss_as_structureinstance(rtssFilepath, contourList, contourList_alt, as_polygon=False):

    rtss = pydicom.read_file(rtssFilepath)

    # Logic for matching structures compared with master list (contourList)
    # TODO implement regex matching for better structure filtering
    structureMatches = []
    for structROI in rtss.StructureSetROISequence:
        structName = structROI.ROIName
        structID = structROI.ROINumber
        structContourSet = None

        # Find contour set in rtss
        for contourSet in rtss.ROIContourSequence:
            if contourSet.ReferencedROINumber == structID:
                structContourSet = contourSet
                break
        if structContourSet is not None:
            if hasattr(structContourSet, 'ContourSequence'):
                numberOfMatches = 0
                lastMatchName = 0
                # print('Checking if structure matches known list: {:s}'.format(structName))

                alt_name_index = 0
                if contourList:
                    for name in contourList:
                        if name.upper() == structName.upper():
                            numberOfMatches = numberOfMatches + 1
                            lastMatchName = name
                        elif contourList_alt[alt_name_index].upper() == structName.upper():
                            numberOfMatches = numberOfMatches + 1
                            lastMatchName = name
                        alt_name_index = alt_name_index + 1

                    if numberOfMatches == 1:
                        structureMatches.append((lastMatchName, structID, structName))
                    elif numberOfMatches == 0:
                        print('\tNo match for structure {:s}\n\tSkipping structure'.format(structName))
                    elif numberOfMatches > 1:
                        print('\tMultiple matches for structure {:s}\n\tSkipping structure'.format(structName))
                else:
                    structureMatches.append((structName, structID, structName))


    # Create python list of polygon dictionaries for each matched structure
    structureData = []
    for idx, matchIDS in enumerate(structureMatches):
        clinicalContourName, structID, structName = matchIDS
        print(clinicalContourName)
        cur_contour_set = None
        for contour_set in rtss.ROIContourSequence:
            if contour_set.ReferencedROINumber == structID:
                cur_contour_set = contour_set
                break

        if as_polygon:
            cur_polygon_tuple = []
            cur_z_slices = []
            if cur_contour_set is not None:
                # get the list of z-values for the reference set
                for cur_contour_slice in cur_contour_set.ContourSequence:
                    n_cur_pts = int(cur_contour_slice.NumberOfContourPoints)
                    if n_cur_pts >= 3:
                        cur_contour = cur_contour_slice.ContourData
                        cur_z_slices.append(cur_contour[2])
                # round to 1 decimal place (0.1mm) to make finding a match more robust
                cur_z_slices = np.round(cur_z_slices, 2)
                cur_z_slices = np.unique(cur_z_slices)

            for z_value in cur_z_slices:
                cur_polygon = None
                for cur_contour_slice in cur_contour_set.ContourSequence:
                    n_cur_pts = int(cur_contour_slice.NumberOfContourPoints)
                    if n_cur_pts >= 3:
                        cur_contour = cur_contour_slice.ContourData
                        if np.round(cur_contour[2], 2) == z_value:
                            # make 2D contours
                            cur_contour_2_d = np.zeros((n_cur_pts, 2))
                            for i in range(0, n_cur_pts):
                                cur_contour_2_d[i][0] = float(cur_contour[i * 3])
                                cur_contour_2_d[i][1] = float(cur_contour[i * 3 + 1])
                            if cur_polygon is None:
                                # Make points into Polygon
                                cur_polygon = Polygon(LinearRing(cur_contour_2_d))
                            else:
                                # Turn next set of points into a Polygon
                                this_cur_polygon = Polygon(LinearRing(cur_contour_2_d))
                                # Attempt to fix any self-intersections in the resulting polygon
                                if not this_cur_polygon.is_valid:
                                    this_cur_polygon = this_cur_polygon.buffer(0)
                                if cur_polygon.contains(this_cur_polygon):
                                    # if the new polygon is inside the old one, chop it out
                                    cur_polygon = cur_polygon.difference(this_cur_polygon)
                                elif cur_polygon.within(this_cur_polygon):
                                    # if the new and vice versa
                                    cur_polygon = this_cur_polygon.difference(cur_polygon)
                                else:
                                    # otherwise it is a floating blob to add
                                    cur_polygon = cur_polygon.union(this_cur_polygon)
                            # Attempt to fix any self-intersections in the resulting polygon
                            if cur_polygon is not None:
                                if not cur_polygon.is_valid:
                                    cur_polygon = cur_polygon.buffer(0)
                cur_polygon_tuple.append([z_value, cur_polygon])
            cur_polygon_tuple = sorted(cur_polygon_tuple, key=getKey)
            structureData.append((clinicalContourName, cur_polygon_tuple))

        else:
            curTuple = []
            if cur_contour_set is not None:
                # get the list of z-values for current contour
                for cur_contour_slice in cur_contour_set.ContourSequence:
                    n_cur_pts = int(cur_contour_slice.NumberOfContourPoints)
                    if n_cur_pts >= 1:
                        curTuple.append([np.float(cur_contour_slice.ContourData[2]), cur_contour_slice.ContourData])
            curTuple = sorted(curTuple, key=getKey)
            if curTuple:
                structureData.append((clinicalContourName, curTuple))

    return structureData, structureMatches


def main():

    print('This file consists of helper functions for DICOM RTSTRUCT QA, visualization and DL Applications')


if __name__ == '__main__':
    main()
