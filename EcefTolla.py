from pyproj import Proj, transform

# Define the projection for ECEF: 'epsg:4978' (ECEF) and WGS84 Latitude/Longitude: 'epsg:4326'
ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = Proj(proj='latlong', ellps='WGS84', datum='WGS84')

def ecef_to_lla(x, y, z):
    lon, lat, alt = transform(ecef, lla, x, y, z, radians=False)
    return lat, lon, alt

