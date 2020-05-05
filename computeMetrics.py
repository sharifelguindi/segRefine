import pickle
from shapely.geometry import *
from shapely.ops import cascaded_union
from shapely.ops import split
import numpy as np
from functions import plot_polygons_and_linestrings, load_arrays_hdf5
import pandas as pd
import os
from skimage import measure
import surface_distance
from PIL import Image, ImageDraw
from operator import itemgetter


def compute_metrics(mask_gt, mask_pred, spacing_mm=(3, 0.6, 0.6), surface_tolerances=[1, 3]):
    surface_DSC = []
    DSC = []
    HD_95 = []
    surface_distances = surface_distance.compute_surface_distances(mask_gt.astype(int),
                                                                   mask_pred.astype(int),
                                                                   spacing_mm=spacing_mm)
    DSC.append(surface_distance.compute_dice_coefficient(mask_gt.astype(int), mask_pred.astype(int)))

    for surface_tolerance in surface_tolerances:
        surface_DSC.append(surface_distance.compute_surface_dice_at_tolerance(surface_distances, surface_tolerance))

    HD_95.append(surface_distance.compute_robust_hausdorff(surface_distances, 95))

    return DSC, surface_DSC, HD_95


def poly2mask(slicePts_ref, pxTransform_inv):

    if slicePts_ref:
        n_cur_pts = int(len(slicePts_ref[0][1]) / 3)
        cur_contour = slicePts_ref[0][1]
        cur_contour_2_d = []
        for i in range(0, n_cur_pts):
            coord = [float(cur_contour[i * 3]), float(cur_contour[i * 3 + 1]), float(cur_contour[i * 3 + 2]), 1]
            pxCoord = np.matmul(pxTransform_inv,coord)
            cur_contour_2_d.append((np.round(pxCoord[0]), np.round(pxCoord[1])))
        img = Image.new('L', (512, 512), 0)
        ImageDraw.Draw(img).polygon(cur_contour_2_d, outline=1, fill=1)
        mask = np.array(img)

    return mask


def mask2poly(mask, slice, pxSize, threshold=0.5):
    contours = measure.find_contours(mask, threshold)
    poly = Polygon()

    for contour in contours:
        pList = []
        if len(contour) > 2 and poly.is_empty == True:
            for point in contour:
                pList.append([point[0] * pxSize[0], point[1] * pxSize[1], slice * pxSize[2]])
            poly = Polygon(pList)
        elif len(contour) > 2 and poly.is_empty == False:
            for point in contour:
                pList.append([point[0] * pxSize[0], point[1] * pxSize[1], slice * pxSize[2]])
            this_ref_polygon = Polygon(pList[:-1])
            if not this_ref_polygon.is_valid:
                this_ref_polygon = this_ref_polygon.buffer(0)

            poly = poly.union(this_ref_polygon)
        else:
            print("mask contained no contour")
    return poly


def get_distance_measures(ref_poly, test_poly, stepsize=1.0, warningsize=1.0):
    # Hausdorff is trivial to compute with Shapely, but average distance requires stepping along each polygon.
    # This is the 'stepsize' in mm. At each point the minimum distance to the other contour is calculated to
    # create a list of distances. From this list the HD can be calculated from this, but it is inaccurate. Therefore,
    # we compare it to the Shapely one and report a problem if the error is greater that 'warningsize' in mm.

    reference_line = ref_poly.boundary
    test_line = test_poly.boundary

    distance_ref_to_test = []
    for distance_along_contour in np.arange(0, reference_line.length, stepsize):
        distance_to_other = reference_line.interpolate(distance_along_contour).distance(test_line)
        distance_ref_to_test.append(distance_to_other)

    distance_test_to_ref = []
    for distance_along_contour in np.arange(0, test_line.length, stepsize):
        distance_to_other = test_line.interpolate(distance_along_contour).distance(reference_line)
        distance_test_to_ref.append(distance_to_other)

    my_hd = np.max([np.max(distance_ref_to_test), np.max(distance_test_to_ref)])
    shapely_hd = test_poly.hausdorff_distance(ref_poly)

    if (my_hd + warningsize < shapely_hd) | (my_hd - warningsize > shapely_hd):
        print('There is a discrepancy between the Hausdorff distance and the list used to calculate the 95% HD')
        print('You may wish to consider a smaller stepsize')

    return distance_ref_to_test, distance_test_to_ref


def get_added_path_length(ref_poly, contracted_poly, expanded_poly, debug=False):
    total_path_length = 0

    reference_boundary = ref_poly.boundary
    if contracted_poly.area > 0:
        contracted_boundary = contracted_poly.boundary
    else:
        contracted_boundary = None
    expanded_boundary = expanded_poly.boundary

    if debug:
        plot_polygons_and_linestrings(reference_boundary, '#000000')
        if contracted_boundary is not None:
            plot_polygons_and_linestrings(contracted_boundary, '#0000ff')
        plot_polygons_and_linestrings(expanded_boundary, '#0000ff')

    if contracted_boundary is not None:
        ref_split_inside = split(reference_boundary, contracted_boundary)
        for line_segment in ref_split_inside:
            # check it the centre of the line is within the contracted polygon
            mid_point = line_segment.interpolate(0.5, True)
            if contracted_poly.contains(mid_point):
                total_path_length = total_path_length + line_segment.length
                if debug:
                    plot_polygons_and_linestrings(line_segment, '#00ff00')
            else:
                if debug:
                    plot_polygons_and_linestrings(line_segment, '#ff0000')

    ref_split_outside = split(reference_boundary, expanded_boundary)
    for line_segment in ref_split_outside:
        # check it the centre of the line is outside the expanded polygon
        mid_point = line_segment.interpolate(0.5, True)
        if not expanded_poly.contains(mid_point):
            total_path_length = total_path_length + line_segment.length
            if debug:
                plot_polygons_and_linestrings(line_segment, '#00ff00')
        else:
            if debug:
                plot_polygons_and_linestrings(line_segment, '#ff0000')

    return total_path_length


def compute_comparison(ref, test, imageSize, sliceLocations, coordTransform):

    total_added_path_length = 0
    total_true_positive_area = 0
    total_false_positive_area = 0
    total_false_negative_area = 0
    total_test_area = 0
    total_ref_area = 0
    distance_ref_to_test = []
    distance_test_to_ref = []
    ref_weighted_centroid_sum = np.array([0, 0, 0])
    test_weighted_centroid_sum = np.array([0, 0, 0])
    tolerance = 1
    max_z = max(max(ref, key=itemgetter(0))[0], max(test, key=itemgetter(0))[0])
    min_z = min(min(ref, key=itemgetter(0))[0], min(test, key=itemgetter(0))[0])

    pxSpacing = [np.round(np.mean(coordTransform[0, 0, :]), 2),
                 np.round(np.mean(coordTransform[1, 1, :]), 2),
                 np.round(np.mean(np.diff(coordTransform[2, 3, :])), 2)]

    mask_ref = np.zeros(imageSize)
    mask_test = np.zeros(imageSize)

    # Loop through min/max in z for comparison of structures
    for z in np.arange(min_z, max_z + pxSpacing[2], pxSpacing[2]):
        z = np.round(z, 2)
        # print(z)
        slicePts_ref  = [item for item in ref if item[0] == z]
        slicePts_test = [item for item in test if item[0] == z]
        pxTransform = coordTransform[:, :, np.where(coordTransform[2, 3, :] == z)][:, :, 0, 0]
        pxTransform_inv = np.linalg.pinv(pxTransform)

        refpolygon = None
        if slicePts_ref:
            if slicePts_ref[0][1]:
                n_cur_pts = int(len(slicePts_ref[0][1]) / 3)
                cur_contour = slicePts_ref[0][1]
                cur_contour_2_d = np.zeros((n_cur_pts, 2))
                for i in range(0, n_cur_pts):
                    cur_contour_2_d[i][0] = float(cur_contour[i * 3])
                    cur_contour_2_d[i][1] = float(cur_contour[i * 3 + 1])
                if refpolygon is None:
                    # Make points into Polygon
                    refpolygon = Polygon(LinearRing(cur_contour_2_d))
                else:
                    # Turn next set of points into a Polygon
                    this_cur_polygon = Polygon(LinearRing(cur_contour_2_d))
                    # Attempt to fix any self-intersections in the resulting polygon
                    if not this_cur_polygon.is_valid:
                        this_cur_polygon = this_cur_polygon.buffer(0)
                    if refpolygon.contains(this_cur_polygon):
                        # if the new polygon is inside the old one, chop it out
                        refpolygon = refpolygon.difference(this_cur_polygon)
                    elif refpolygon.within(this_cur_polygon):
                        # if the new and vice versa
                        refpolygon = this_cur_polygon.difference(refpolygon)
                    else:
                        # otherwise it is a floating blob to add
                        refpolygon = refpolygon.union(this_cur_polygon)
                # Attempt to fix any self-intersections in the resulting polygon
                if refpolygon is not None:
                    if not refpolygon.is_valid:
                        refpolygon = refpolygon.buffer(0)
                mask_ref[np.where(sliceLocations == z), :, :, 0] = poly2mask(slicePts_ref, pxTransform_inv)

        testpolygon = None
        if slicePts_test:
            if slicePts_test[0][1]:
                n_cur_pts = int(len(slicePts_test[0][1]) / 3)
                cur_contour = slicePts_test[0][1]
                cur_contour_2_d = np.zeros((n_cur_pts, 2))
                for i in range(0, n_cur_pts):
                    cur_contour_2_d[i][0] = float(cur_contour[i * 3])
                    cur_contour_2_d[i][1] = float(cur_contour[i * 3 + 1])
                if testpolygon is None:
                    # Make points into Polygon
                    testpolygon = Polygon(LinearRing(cur_contour_2_d))
                else:
                    # Turn next set of points into a Polygon
                    this_cur_polygon = Polygon(LinearRing(cur_contour_2_d))
                    # Attempt to fix any self-intersections in the resulting polygon
                    if not this_cur_polygon.is_valid:
                        this_cur_polygon = this_cur_polygon.buffer(0)
                    if testpolygon.contains(this_cur_polygon):
                        # if the new polygon is inside the old one, chop it out
                        testpolygon = testpolygon.difference(this_cur_polygon)
                    elif testpolygon.within(this_cur_polygon):
                        # if the new and vice versa
                        testpolygon = this_cur_polygon.difference(testpolygon)
                    else:
                        # otherwise it is a floating blob to add
                        testpolygon = testpolygon.union(this_cur_polygon)
                # Attempt to fix any self-intersections in the resulting polygon
                if testpolygon is not None:
                    if not testpolygon.is_valid:
                        testpolygon = testpolygon.buffer(0)
                mask_test[np.where(sliceLocations == z), :, :, 0] = poly2mask(slicePts_test, pxTransform_inv)

        if refpolygon is not None and testpolygon is not None:

            # go get some distance measures
            # these get added to a big list so that we can calculate the 95% HD
            [ref_to_test, test_to_ref] = get_distance_measures(refpolygon, testpolygon, 0.05)
            distance_ref_to_test.extend(ref_to_test)
            distance_test_to_ref.extend(test_to_ref)

            # apply tolerance ring margin to test with added path length
            expanded_poly = cascaded_union(testpolygon.buffer(tolerance, 32, 1, 1))
            contracted_poly = cascaded_union(testpolygon.buffer(-tolerance, 32, 1, 1))

            # add intersection of contours
            contour_intersection = refpolygon.intersection(testpolygon)
            total_true_positive_area = total_true_positive_area + contour_intersection.area
            total_false_negative_area = total_false_negative_area + \
                                        (refpolygon.difference(contour_intersection)).area
            total_false_positive_area = total_false_positive_area + \
                                        (testpolygon.difference(contour_intersection)).area
            total_test_area = total_test_area + testpolygon.area
            total_ref_area = total_ref_area + refpolygon.area
            centroid_point = refpolygon.centroid
            centroid_point_np = np.array([centroid_point.x, centroid_point.y, z])
            ref_weighted_centroid_sum = ref_weighted_centroid_sum + (refpolygon.area * centroid_point_np)
            centroid_point = testpolygon.centroid
            centroid_point_np = np.array([centroid_point.x, centroid_point.y, z])
            test_weighted_centroid_sum = test_weighted_centroid_sum + (testpolygon.area * centroid_point_np)

            # add length of remain contours

            added_path = get_added_path_length(refpolygon, contracted_poly, expanded_poly)
            total_added_path_length = total_added_path_length + added_path

        elif refpolygon is not None and testpolygon is None:
            # if no corresponding slice, then add the whole ref length
            # print('Adding path for whole contour')
            path_length = refpolygon.length
            total_added_path_length = total_added_path_length + path_length
            # also the whole slice is false negative
            total_false_negative_area = total_false_negative_area + refpolygon.area
            total_ref_area = total_ref_area + refpolygon.area
            centroid_point = refpolygon.centroid
            centroid_point_np = np.array([centroid_point.x, centroid_point.y, z])
            ref_weighted_centroid_sum = ref_weighted_centroid_sum + (refpolygon.area * centroid_point_np)

        elif refpolygon is None and testpolygon is not None:
            total_false_positive_area = total_false_positive_area + testpolygon.area
            total_test_area = total_test_area + testpolygon.area
            centroid_point = testpolygon.centroid
            centroid_point_np = np.array([centroid_point.x, centroid_point.y, z])
            test_weighted_centroid_sum = test_weighted_centroid_sum + (testpolygon.area * centroid_point_np)

    if not (not testpolygon and not refpolygon):
        ref_centroid = ref_weighted_centroid_sum / total_ref_area
        test_centroid = test_weighted_centroid_sum / total_ref_area

        if len(distance_ref_to_test) > 1:
            hd = np.max([np.max(distance_ref_to_test), np.max(distance_test_to_ref)])
            ninety_five_hd = np.max(
                [np.percentile(distance_ref_to_test, 95), np.percentile(distance_test_to_ref, 95)])
            ave_dist = (np.mean(distance_ref_to_test) + np.mean(distance_test_to_ref)) / 2
            median_dist = (np.median(distance_ref_to_test) + np.median(distance_test_to_ref)) / 2
        else:
            hd = 0
            ninety_five_hd = 0
            ave_dist = 0
            median_dist = 0

        tau = [1, 3]
        VDSC, surface_DSC, HD_95 = compute_metrics(mask_ref[:, :, :, 0], mask_test[:, :, :, 0],
                                                   spacing_mm=pxSpacing, surface_tolerances=tau)
        SEN = total_true_positive_area / total_ref_area
        pctFP = total_false_positive_area / total_test_area
        dsc = (2 * total_true_positive_area) / (total_ref_area + total_test_area)
        results = [total_added_path_length, total_true_positive_area * pxSpacing[2],
                   total_false_negative_area * pxSpacing[2],
                   total_false_positive_area * pxSpacing[2], SEN,
                   pctFP, dsc, hd, ninety_five_hd, ave_dist, median_dist,
                   ref_centroid, test_centroid, surface_DSC[0], surface_DSC[1],
                   HD_95, total_ref_area * pxSpacing[2], total_test_area * pxSpacing[2]]
    else:
        # Return Null if both polygons are empty (in case of error)
        results = [a*0 for a in range(0, 19)]

    return results, mask_ref - mask_test


def main():

    masterStructureList = "G:\\Projects\\mimTemplates\\StructureListMaster.xlsx"
    structureList = pd.read_excel(masterStructureList)
    HDF5_DIR = "H:\\Treatment Planning\\Elguindi\\storage"
    contourDatabase = "H:\\Treatment Planning\\Elguindi\\contourDatabase\\contourDB.xlsx"
    db = pd.read_excel(contourDatabase, index=False)

    # toCompare = [testContour, refContour]
    toCompare = ['autoGenerated', 'mimLatest']

    comparisonMetrics = ['APL', 'TP volume', 'FN volume', 'FP volume', 'SEN', '%FP',
                         '3D DSC', '2D HD', '95% 2D HD', 'Ave 2D Dist', 'Median 2D Dist',
                         'Reference Centroid', 'Test Centroid', 'SDSC_1mm', 'SDSC_3mm',
                         'RobustHD_95', 'ref_vol', 'test_vol']

    for id in db['MRN']:
        rowVal = db.index[db['MRN'] == id].tolist()[0]
        sl = [x.upper() for x in structureList['StructureName'].to_list()]
        if rowVal >= 0:
            print(id)
            data = db[db['MRN'] == id].dropna(axis=1)
            scanData = load_arrays_hdf5(HDF5_DIR, data['SCAN_FILE'].iloc[0])
            coordTransform = scanData['coordinateSystemTransform']
            imageSize = scanData['imageSize']
            sliceLocations = scanData['sliceLocations']

            for col in data.columns:
                for s in sl:
                    if s in col:
                        organname = s
                        sl.remove(s)
                        if organname + toCompare[1] in data:
                            ref = pickle.load(
                                open(os.path.join(HDF5_DIR, data[organname + toCompare[1]].iloc[0]), 'rb'))
                        else:
                            ref = []

                        if organname + toCompare[0] in data:
                            test = pickle.load(open(os.path.join(HDF5_DIR, data[organname + toCompare[0]].iloc[0]), 'rb'))
                        else:
                            test = []

                        if ref and not test:
                            test = ref
                            for k in range(0, len(test)):
                                test[k] = (test[k][0], [])
                        if test and not ref:
                            ref = test
                            for k in range(0, len(ref)):
                                ref[k] = (ref[k][0], [])

                        if ref and test:
                            print('Computing metrics for organ: ' + organname)
                            scores, mask_ref = compute_comparison(ref, test, imageSize, sliceLocations, coordTransform)
                            results_structures = {}
                            k = 0
                            for metric in comparisonMetrics:
                                metricName = organname + '_' + metric
                                results_structures[metricName] = str(scores[k]).strip('[').strip(']').lstrip().replace(' ','_')
                                k = k + 1

                            dbRow = pd.Series(results_structures)
                            db.loc[rowVal, dbRow.index] = dbRow
        print('done')
    db.to_excel(contourDatabase, index=False)

    return


if __name__ == '__main__':
    main()