import pydicom
import numpy as np
from functions import getKey
from computeMetrics import compute_comparison
import ast

# Input is list of all dcm Image file locations
def storeCTSeries(dcmFiles):

    arrayTuple = []
    for dcmFile in dcmFiles:
        dataset = pydicom.dcmread(dcmFile)
        pixelSpacing = dataset.PixelSpacing
        pixelSpacing.append(dataset.SliceThickness)
        ImagePosition = dataset.ImagePositionPatient
        ImagePosition.append(1)
        X = dataset.ImageOrientationPatient[0:3]
        Y = dataset.ImageOrientationPatient[3:]
        coordinateSystemTransform = np.zeros((4, 4))
        coordinateSystemTransform[0:3, 0] = np.asarray(X) * np.asarray(pixelSpacing[0])
        coordinateSystemTransform[0:3, 1] = np.asarray(Y) * np.asarray(pixelSpacing[1])
        coordinateSystemTransform[0:, 3] = ImagePosition
        arrayTuple.append([np.float(dataset.SliceLocation), dataset.pixel_array, coordinateSystemTransform])

    arrayTuple = sorted(arrayTuple, key=getKey)
    data = {}
    sliceLocations = np.zeros((len(arrayTuple)))
    imageSize = np.asarray([len(arrayTuple), dataset.Rows, dataset.Columns, 1])
    pixelConversion = np.asarray([dataset.RescaleIntercept, dataset.RescaleSlope])
    pixelData = np.zeros(imageSize)
    coordinateTransforms = np.zeros((4, 4, len(arrayTuple)))

    l = 0
    for sliceLocation, image, coordinateSystemTransform in arrayTuple:
        coordinateTransforms[:, :, l] = coordinateSystemTransform
        sliceLocations[l] = sliceLocation
        pixelData[l, :, :, 0] = image
        l = l + 1
    data['sliceLocations'] = sliceLocations
    data['imageSize'] = imageSize
    data['pixelConversion'] = pixelConversion
    data['imageMatrix'] = pixelData
    data['coordinateSystemTransform'] = coordinateTransforms

    return data

# single instance of structure, only 'latestMimInstance' and 'latestEclipseInstance' and output from storedCTSeries
def compareContours(ref_str, test_str, scanData):

    comparisonMetrics = ['APL', 'TP volume', 'FN volume', 'FP volume', 'SEN', '%FP',
                         '3D DSC', '2D HD', '95% 2D HD', 'Ave 2D Dist', 'Median 2D Dist',
                         'Reference Centroid', 'Test Centroid', 'SDSC_1mm', 'SDSC_3mm',
                         'RobustHD_95', 'ref_vol', 'test_vol']

    coordTransform = scanData['coordinateSystemTransform']
    imageSize = scanData['imageSize']
    sliceLocations = scanData['sliceLocations']

    ref_raw = ast.literal_eval(ref_str)
    test_raw = ast.literal_eval(test_str)

    ref = []
    for key, value in ref_raw:
        if len(value) >= 1:
            ref.append([np.float(value[2]), value])
        ref = sorted(ref, key=getKey)

    test = []
    for key, value in test_raw:
        if len(value) >= 1:
            ref.append([np.float(value[2]), value])
        test = sorted(test, key=getKey)

    scores, mask_ref = compute_comparison(ref, test, imageSize, sliceLocations, coordTransform)
    results_structures = {}
    k = 0
    for metric in comparisonMetrics:
        results_structures[metric] = str(scores[k]).strip('[').strip(']').lstrip().replace(' ', '_')
        k = k + 1

    return results_structures
