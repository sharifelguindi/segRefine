import pydicom
import ast
from shapely.geometry import *
from shapely.ops import cascaded_union
from shapely.ops import split
import numpy as np
from skimage import measure
import surface_distance
from PIL import Image, ImageDraw
from operator import itemgetter
import matplotlib.pyplot as plt


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


def getKey(item):
    return item[0]


def compute_metrics(mask_gt, mask_pred, spacing_mm=(3, 0.6, 0.6), surface_tolerances=None):
    if surface_tolerances is None:
        surface_tolerances = [1, 3]
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


def poly2mask(slicePts_ref, pxTransform_inv, imageSize):
    if slicePts_ref:
        n_cur_pts = int(len(slicePts_ref[0][1]) / 3)
        cur_contour = slicePts_ref[0][1]
        cur_contour_2_d = []
        for i in range(0, n_cur_pts):
            coord = [float(cur_contour[i * 3]), float(cur_contour[i * 3 + 1]), float(cur_contour[i * 3 + 2]), 1]
            pxCoord = np.matmul(pxTransform_inv, coord)
            cur_contour_2_d.append((np.round(pxCoord[0]), np.round(pxCoord[1])))
        img = Image.new('L', (imageSize[1], imageSize[2]), 0)
        ImageDraw.Draw(img).polygon(cur_contour_2_d, outline=1, fill=1)
        mask = np.array(img)
    else:
        mask = np.asarray(Image.new('L', (imageSize[1], imageSize[2]), 0))

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


# noinspection PyUnresolvedReferences
def compute_comparison(ref, test, imageSize, sliceLocations, coordTransform):
    rndDig = 3
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

    pxSpacing = [np.round(np.mean(coordTransform[0, 0, :]), rndDig),
                 np.round(np.mean(coordTransform[1, 1, :]), rndDig),
                 np.round(np.mean(np.diff(coordTransform[2, 3, :])), rndDig)]

    mask_ref = np.zeros(imageSize)
    mask_test = np.zeros(imageSize)

    # Loop through min/max in z for comparison of structures
    for z in np.arange(min_z, max_z + pxSpacing[2], pxSpacing[2]):
        rndDig = 3
        z = np.round(z, rndDig)

        if z <= sliceLocations[-1]:
            slicePts_ref = [item for item in ref if item[0] == z]
            slicePts_test = [item for item in test if item[0] == z]
            zLocValue = min(coordTransform[2, 3, :], key=lambda x: abs(x - z))
            zLoc = np.where(coordTransform[2, 3, :] == zLocValue)
            pxTransform = coordTransform[:, :, zLoc[0][0]]

            # # Verify z location through continual rounding for robustness
            # if zLoc[0]:
            #     pxTransform = coordTransform[:, :, zLoc[0][0]]
            # else:
            #     rndDig = 2
            #     z = np.round(z, rndDig)
            #     zLoc = np.where(np.round(coordTransform[2, 3, :], rndDig) == z)
            #     if zLoc[0]:
            #         pxTransform = coordTransform[:, :, zLoc[0][0]]
            #     else:
            #         rndDig = 1
            #         z = np.round(z, rndDig)
            #         zLoc = np.where(np.round(coordTransform[2, 3, :], rndDig) == z)
            #         if zLoc[0]:
            #             pxTransform = coordTransform[:, :, zLoc[0][0]]
            #         else:
            #             zLoc = np.where(np.round(coordTransform[2, 3, :]) == z)
            #             pxTransform = coordTransform[:, :, zLoc[0][0]]

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
                    mask_ref[np.where(np.round(sliceLocations, rndDig) == z), :, :, 0] = poly2mask(slicePts_ref,
                                                                                                   pxTransform_inv,
                                                                                                   imageSize)

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
                    mask_test[np.where(np.round(sliceLocations, rndDig) == z), :, :, 0] = poly2mask(slicePts_test,
                                                                                                    pxTransform_inv,
                                                                                                    imageSize)

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
    if total_ref_area > 0:
        SEN = total_true_positive_area / total_ref_area
    else:
        SEN = 0

    if total_test_area > 0:
        pctFP = total_false_positive_area / total_test_area
    else:
        pctFP = 0

    if total_test_area > 0 and total_ref_area > 0:
        dsc = (2 * total_true_positive_area) / (total_ref_area + total_test_area)
    else:
        dsc = 0

    results = [total_added_path_length, total_true_positive_area * pxSpacing[2],
               total_false_negative_area * pxSpacing[2],
               total_false_positive_area * pxSpacing[2], SEN,
               pctFP, dsc, hd, ninety_five_hd, ave_dist, median_dist,
               ref_centroid, test_centroid, surface_DSC[0], surface_DSC[1],
               HD_95, total_ref_area * pxSpacing[2], total_test_area * pxSpacing[2]]

    return results, mask_ref, mask_test


def store_rtss_as_structureinstance(rtssFilepath, contourList, contourList_alt, contourList_var, as_polygon=False):
    rtss = pydicom.read_file(rtssFilepath)
    rndDig = 3
    # Logic for matching structures compared with master list (contourList)
    # TODO implement regex matching for better structure filtering
    structureMatches = []
    for structROI in rtss.StructureSetROISequence:
        structName = structROI.ROIName
        structID = structROI.ROINumber
        structContourSet = None

        # Find contour set in rtss
        for contourSet in rtss.ROIContourSequence:
            try:
                if contourSet.ReferencedROINumber == structID:
                    structContourSet = contourSet
                    break
            except AttributeError:
                print("ReferenceROI not found")

        if structContourSet is not None:
            if hasattr(structContourSet, 'ContourSequence'):
                numberOfMatches = 0
                lastMatchName = 0
                alt_name_index = 0

                if structName.upper()[-2:] == '_F':
                    structName = structName[0:-2]
                if structName.upper()[-1] == '1':
                    if 'FIDUCIAL' in structName.upper():
                        structName = structName
                    else:
                        structName = structName[0:-1]
                if contourList:
                    for name in contourList:
                        if name.upper() == structName.upper():
                            numberOfMatches = numberOfMatches + 1
                            lastMatchName = name
                        elif contourList_alt[alt_name_index].upper() == structName.upper():
                            numberOfMatches = numberOfMatches + 1
                            lastMatchName = name
                        elif contourList_var[alt_name_index].upper() == structName.upper():
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
                cur_z_slices = np.round(cur_z_slices, rndDig)
                cur_z_slices = np.unique(cur_z_slices)

            for z_value in cur_z_slices:
                cur_polygon = None
                for cur_contour_slice in cur_contour_set.ContourSequence:
                    n_cur_pts = int(cur_contour_slice.NumberOfContourPoints)
                    if n_cur_pts >= 3:
                        cur_contour = cur_contour_slice.ContourData
                        if np.round(cur_contour[2], rndDig) == z_value:
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


# Input is list of all dcm Image file locations, from treatment summary json you can generate list like this:

# dcmFiles = []
# for item in summary['courses'][1]['4ContourQA']['plans'][0]['4ContourQA']['CT_SERIES']:
#     dcmFiles.append(item['dicomLocation'])
#
# This generates the DICOM files needed for the plan under course index '1' and plan index '0'.  The CT SERIES has an
# associated Series UID that can link it to the appropriate structure instance json file

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


# Inputs: data returned from function storeCTSeries and 2 structure_instance strings.
# You can generate ref_str and test_str like this:
# ref_str = f[0]['latestMimInstance']
# test_str = f[0]['latestEclipseInstance']
# where f is the json file for the structure instance of a patient and the comparison is done on the 1st structure found
# the result is a python dict with the values computed


def compareContours(ref_str, test_str, scanData):
    comparisonMetrics = ['APL', 'TP volume', 'FN volume', 'FP volume', 'SEN', '%FP',
                         '3D DSC', '2D HD', '95% 2D HD', 'Ave 2D Dist', 'Median 2D Dist',
                         'Reference Centroid', 'Test Centroid', 'SDSC_1mm', 'SDSC_3mm',
                         'RobustHD_95', 'ref_vol', 'test_vol']
    rndDig = 3
    coordTransform = scanData['coordinateSystemTransform']
    imageSize = scanData['imageSize']
    sliceLocations = scanData['sliceLocations']
    sliceLocations = np.round(sliceLocations, rndDig)
    ref_raw = ast.literal_eval(ref_str)
    test_raw = ast.literal_eval(test_str)

    ref = []
    for key, value in ref_raw.items():
        if len(value) >= 9:
            ref.append([np.float(value[2]), value])
        ref = sorted(ref, key=getKey)

    test = []
    for key, value in test_raw.items():
        if len(value) >= 9:
            test.append([np.float(value[2]), value])
        test = sorted(test, key=getKey)

    if ref and test:
        scores, mask_ref, mask_test = compute_comparison(ref, test, imageSize, sliceLocations, coordTransform)
        results_structures = {}
        k = 0
        for metric in comparisonMetrics:
            results_structures[metric] = str(scores[k]).strip('[').strip(']').lstrip().replace(' ', '_')
            k = k + 1
    else:
        results_structures = {}

    return results_structures
