from functions import find, store_arrays_hdf5
from sparkFunctions import store_rtss_as_structureinstance, getKey
import pydicom
import os
import numpy as np
import pandas as pd
from compress_pickle import dump
from datetime import datetime


HDF5_DIR = "H:\\Treatment Planning\\Elguindi\\storage"
masterStructureList = "G:\\Projects\\mimTemplates\\StructureListMaster.xlsx"
contourDatabase = "H:\\Treatment Planning\\Elguindi\\contourDatabase\\contourDB.xlsx"

structureTypes = ['autoGenerated', 'mimLatest', 'physicianApproved', 'eclipseLatest',
                  'APL', 'TP volume', 'FN volume', 'FP volume', 'SEN', '%FP',
                  '3D DSC', '2D HD', '95% 2D HD', 'Ave 2D Dist', 'Median 2D Dist',
                  'Reference Centroid', 'Test Centroid', 'SDSC_1mm', 'SDSC_3mm', 'RobustHD_95', 'ref_vol', 'test_vol']

keywordMatch = {'autoGenerated': 'ATLAS'}
structureList = pd.read_excel(masterStructureList)
sl = [x.upper() for x in structureList['StructureName'].to_list()]
sl_alt = [str(x).upper() for x in structureList['TG263_Equivalent'].to_list()]

# Generate structure data columns based on types/names
columns = ['MRN', 'SCAN_FILE', 'SCAN_DATE']
for struct in sl:
    for structureType in structureTypes:
        columns.append(struct + '_' + structureType)

# find DCM directories to process
dcmDirectory = 'H:\\Treatment Planning\\Elguindi\\prostateSegAnalysis\\dicomStorage\\'
dirList = [x[0] for x in os.walk(dcmDirectory)]

# create pandas database, contourDatabase non-existent
if not os.path.isfile(contourDatabase):
    db = pd.DataFrame(columns=columns, dtype='str')
else:
    print('Database Exists, adding patient study: ' + contourDatabase)
    db = pd.read_excel(contourDatabase, index=False)

for column in columns:
    if column not in db:
        db[column] = ""
    else:
        db[column] = db[column].astype('str')


for directory in dirList:

    # if directory is CT, store scanData matrix
    print(directory.split(os.sep)[-1][0:2])
    structureSets = []
    dataset_ct = None
    if 'CT' in directory.split(os.sep)[-1][0:2]:
        dcmFiles = find('*.dcm', directory)
        if dcmFiles:
            arrayTuple = []
            for dcmFile in dcmFiles:
                dataset = pydicom.dcmread(dcmFile)
                filename = dataset.StudyInstanceUID
                file = os.path.join(HDF5_DIR, filename + '.h5')
                if not os.path.isfile(file) or os.path.isfile(file):
                    if dataset.Modality == 'CT':
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
                        dataset_ct = dcmFile
                    elif dataset.Modality == 'RTSTRUCT':
                        print('found RTSTRUCT')
                        date_obj = datetime.strptime(dataset.InstanceCreationDate + dataset.InstanceCreationTime[0:5], '%Y%m%d%H%M%S')
                        structureSets.append((date_obj, dcmFile))

            if dataset_ct:
                dataset = pydicom.dcmread(dataset_ct)
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
                date_obj = datetime.strptime(dataset.StudyDate + dataset.StudyTime[0:5], '%Y%m%d%H%M%S')
                filename = dataset.StudyInstanceUID
                file = os.path.join(HDF5_DIR, filename + '.h5')
                if not (db['MRN'] == dataset.PatientID).any():
                    db = db.append({'MRN': dataset.PatientID}, ignore_index=True)
                else:
                    print('Patient is in database')
                if not os.path.isfile(file):
                    store_arrays_hdf5(data, HDF5_DIR, filename)
                    if not (db['SCAN_FILE'] == filename + '.h5').any():
                        db.at[db.index[db['MRN'] == dataset.PatientID].tolist()[0], 'SCAN_FILE'] = filename + '.h5'
                        db.at[db.index[db['MRN'] == dataset.PatientID].tolist()[0], 'SCAN_DATE'] = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        print('Series already in database')
                else:
                    print('Same Image Series UID Exists: ' + filename)

                structureSets = sorted(structureSets, key=getKey)

                if len(structureSets) == 2:
                    dcmFile = structureSets[0][1]
                    dataset = pydicom.dcmread(dcmFile)
                    timeStamp = dataset.StructureSetDate + '-' + dataset.StructureSetTime.replace('.', '-')
                    if not (db['MRN'] == dataset.PatientID).any():
                        db = db.append({'MRN': dataset.PatientID}, ignore_index=True)
                    print('getting structures in file: ' + dataset.SeriesDescription)
                    structureData, structureMatches = store_rtss_as_structureinstance(dcmFile, sl, sl_alt,
                                                                                      as_polygon=False)
                    for structure in structureData:
                        filename = structure[0] + '-' + timeStamp + '.gz'
                        dump(structure[1], open(os.path.join(HDF5_DIR, filename), 'wb'), compression='gzip')
                        column_name = structure[0].upper() + '_' + structureTypes[0]
                        if not (db[column_name] == filename).any():
                            db.at[db.index[db['MRN'] == dataset.PatientID].tolist()[0], column_name] = filename
                        else:
                            print('Structure Already in Database:' + structure[0])
                    print('...')
                    print('Row updated for patient directory: ' + directory)
                    db.to_excel(contourDatabase, index=False)

                    dcmFile = structureSets[1][1]
                    dataset = pydicom.dcmread(dcmFile)
                    timeStamp = dataset.StructureSetDate + '-' + dataset.StructureSetTime.replace('.', '-')
                    if not (db['MRN'] == dataset.PatientID).any():
                        db = db.append({'MRN': dataset.PatientID}, ignore_index=True)
                    print('getting structures in file: ' +dataset.SeriesDescription)
                    structureData, structureMatches = store_rtss_as_structureinstance(dcmFile, sl, sl_alt,
                                                                                      as_polygon=False)
                    for structure in structureData:
                        filename = structure[0] + '-' + timeStamp + '.gz'
                        dump(structure[1], open(os.path.join(HDF5_DIR, filename), 'wb'), compression='gzip')
                        column_name = structure[0].upper() + '_' + structureTypes[1]
                        if not (db[column_name] == filename).any():
                            db.at[db.index[db['MRN'] == dataset.PatientID].tolist()[0], column_name] = filename
                        else:
                            print('Structure Already in Database:' + structure[0])
                    print('...')
                    print('Row updated for patient directory: ' + directory)
                    db.to_excel(contourDatabase, index=False)

    # # if RTSTRUCT file, store based on defined tag
    # elif 'RTSTRUCT' in directory.upper() and 'AAAAATLAS' in directory.upper():
    #
    #     dcmFiles = find('*.dcm', os.path.join(triggerFile.strip('.trigger'), directory))
    #     print(directory)
    #     if len(dcmFiles) == 1:
    #         dataset = pydicom.dcmread(dcmFiles[0])
    #         timeStamp = dataset.StructureSetDate + '-' + dataset.StructureSetTime.replace('.', '-')
    #         if not (db['MRN'] == dataset.PatientID).any():
    #             db = db.append({'MRN': dataset.PatientID}, ignore_index=True)
    #         structureData, structureMatches = store_rtss_as_structureinstance(dcmFiles[0], sl, sl_alt,
    #                                                                           as_polygon=False)
    #         for structure in structureData:
    #             filename = structure[0] + '-' + timeStamp + '.gz'
    #             dump(structure[1], open(os.path.join(HDF5_DIR, filename), 'wb'), compression='gzip')
    #             column_name = structure[0].upper() + '_' + structureTypes[0]
    #             if not (db[column_name] == filename).any():
    #                 db.at[db.index[db['MRN'] == dataset.PatientID].tolist()[0], column_name] = filename
    #             else:
    #                 print('Structure Already in Database:' + structure[0])
    #     else:
    #         print('Multiple RTSTRUCT files labled the same in directory: ' + directory)
    #     print('...')
    #     print('Row updated for patient directory: ' + directory)
    #     db.to_excel(contourDatabase, index=False)
    #
    # elif 'RTSTRUCT' in directory.upper() and 'BBBBB' in directory.upper():
    #
    #     dcmFiles = find('*.dcm', os.path.join(triggerFile.strip('.trigger'), directory))
    #     print(directory)
    #     if len(dcmFiles) == 1:
    #         dataset = pydicom.dcmread(dcmFiles[0])
    #         timeStamp = dataset.StructureSetDate + '-' + dataset.StructureSetTime.replace('.', '-')
    #         if not (db['MRN'] == dataset.PatientID).any():
    #             db = db.append({'MRN': dataset.PatientID}, ignore_index=True)
    #         structureData, structureMatches = store_rtss_as_structureinstance(dcmFiles[0], sl, sl_alt,
    #                                                                           as_polygon=False)
    #         for structure in structureData:
    #             filename = structure[0] + '-' + timeStamp + '.gz'
    #             dump(structure[1], open(os.path.join(HDF5_DIR, filename), 'wb'), compression='gzip')
    #             column_name = structure[0].upper() + '_' + structureTypes[1]
    #             if not (db[column_name] == filename).any():
    #                 db.at[db.index[db['MRN'] == dataset.PatientID].tolist()[0], column_name] = filename
    #             else:
    #                 print('Structure Already in Database:' + structure[0])
    #     else:
    #         print('Multiple RTSTRUCT files labled the same in directory: ' + directory)
    #     print('...')
    #     print('Row updated for patient directory: ' + directory)
    #     db.to_excel(contourDatabase, index=False)





