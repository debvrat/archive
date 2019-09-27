import gdal
import numpy as np
import pandas as pd


def predictClassMap(outRaster, rasterDS, classifier):
	geo_transform = rasterDS.GetGeoTransform()
	projection = rasterDS.GetProjectionRef()
	bandsData = bandDatatoNumpy(rasterDS)
	rows, cols, noBands = bandsData.shape

	result = classifier.predict(bandsData.reshape(rows*cols, noBands))
	classification = result.reshape((rows, cols))
	createGeotiff(outRaster, classification, geo_transform, projection)

def createGeotiff(outRaster, data, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    rasterDS = driver.Create(outRaster, cols, rows, 1, gdal.GDT_Byte)
    rasterDS.SetGeoTransform(geo_transform)
    rasterDS.SetProjection(projection)
    band = rasterDS.GetRasterBand(1)
    band.WriteArray(data)
    dataset = None

def bandDatatoNumpy(rasterDS):
	bandsData = []
	for b in range(rasterDS.RasterCount):
	    band = rasterDS.GetRasterBand(b+1)
	    band_arr = band.ReadAsArray()
	    bandsData.append(band_arr)
	bandsData = np.dstack(bandsData)
	return bandsData

def metricCalc(confMat):
	pa_c = confMat[0,0] / np.sum(confMat[:,0])
	ua_c = confMat[0,0] / np.sum(confMat[0,:]) 
	pa_sn = confMat[1,1] / np.sum(confMat[:,1])
	ua_sn = confMat[1,1] / np.sum(confMat[1,:])
	pa_sh = confMat[2,2] / np.sum(confMat[:,2])
	ua_sh = confMat[2,2] / np.sum(confMat[2,:]) 
	pa_r = confMat[3,3] / np.sum(confMat[:,3])
	ua_r = confMat[3,3] / np.sum(confMat[3,:]) 
	f_c = stats.hmean([pa_c, ua_c])
	f_sn = stats.hmean([pa_sn, ua_sn])
	f_2cls = np.mean([f_c, f_sn])
	oa = np.trace(confMat)/np.sum(confMat)
	return pa_c*100, ua_c*100, pa_sn*100, ua_sn*100, pa_sh*100, ua_sh*100, pa_r*100, ua_r*100, f_c*100, f_sn*100, f_2cls*100, oa*100


def writeToExcel(testVec, trainVec, filename):
	df = pd.DataFrame({'Test Data': np.array(testVec), 'Train Data': np.array(trainVec)})
	writer = pd.ExcelWriter(filename+'.xlsx', engine='xlsxwriter')
	df.to_excel(writer)
	writer.save()