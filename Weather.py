import geopandas as gpd

shapefile_path = r"D:\OneDrive - University of Cambridge\Tripos\Projects\Weather\Data\Köppen_Geiger_1976_2000\c1976_2000.shp"
gdf = gpd.read_file(shapefile_path)

print(gdf.head())


import matplotlib.pyplot as plt

gdf.plot()
plt.show()
