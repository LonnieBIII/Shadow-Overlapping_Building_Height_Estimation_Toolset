"""

Batch Shadow Detection tool
---------------------------

Purpose: For given RGB image, performs shadow detection using eight different algorithms

Author:
- Lonnie Byrnside III

Additional Notes:
- Citations for original papers and code implementations available for each algorithm in respective .py file
- Spatial reference management function used in each algorithm pulled from:
    Schuyler Erle, Frank Warmerdam, Even Rouault
    https://svn.osgeo.org/gdal/trunk/gdal/swig/python/samples/gdalcopyproj.py
    2005

"""

import os

import shadow_detection_Arevalo
import shadow_detection_Deb_and_Suny
import shadow_detection_Freitas
import shadow_detection_Ma
import shadow_detection_Murali
import shadow_detection_Silva
import shadow_detection_Singh
import shadow_detection_Tsai



input_image = input("Enter image path: ")
folder_path = os.path.dirname(input_image)
base_name = os.path.splitext(input_image)[0]



shadow_detection_Arevalo.main(input_image, folder_path, base_name)
shadow_detection_Deb_and_Suny.main(input_image, folder_path, base_name)
shadow_detection_Freitas.main(input_image, folder_path, base_name)
shadow_detection_Ma.main(input_image, folder_path, base_name)
shadow_detection_Murali.main(input_image, folder_path, base_name)
shadow_detection_Silva.main(input_image, folder_path, base_name)
shadow_detection_Singh.main(input_image, folder_path, base_name)
shadow_detection_Tsai.main(input_image, folder_path, base_name)


print("Complete!")
