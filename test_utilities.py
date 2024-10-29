import geopandas as gpd
import numpy as np
import shapely
import hashlib

from sampler import Dataset_patches

def create_centroid_identifier(
    bf_dataset: gpd.GeoDataFrame,
    id_column_name: str = "object_id"
) -> gpd.GeoDataFrame:
    # Calculate centroid for each building polygon
    centroids = bf_dataset.centroid

    # Generate a unique ID for each building based on the hash sum of lat and lon of the centroid
    def hash_sum(x):
        # Create a hash for the latitude and another for the longitude
        hash_lat = hashlib.sha256(str(x.y).encode()).hexdigest()
        hash_lon = hashlib.sha256(str(x.x).encode()).hexdigest()
        # Sum the two hashes
        return hash_lat + hash_lon

    bf_dataset[id_column_name] = centroids.apply(hash_sum)
    return bf_dataset


def get_random_gdf(num_records: int):
    central_point = shapely.geometry.Point(np.random.uniform(-50, 50), np.random.uniform(-50, 50))
    points = [
        shapely.geometry.Point(
            central_point.x + np.random.uniform(-1, 1),
            central_point.y + np.random.uniform(-1, 1)
        )
        for _ in range(num_records)
    ]

    # Generate GeoDataFrame
    return gpd.GeoDataFrame(
        {"object_id": list(range(num_records))},
        geometry=points,
        crs="EPSG:4326"
    )


def test_gdf_mask():
    mask = np.zeros(10, 10, dtype=int)
    mask[7:10, 7:10] = 1

    gdf = get_random_gdf(10)
    gdf["box_pixel"] = shapely.geometry.box(7, 7, 10, 10)
    gdf.iloc[0, gdf.columns.get_loc("box_pixel")] = shapely.geometry.box(0, 0, 3, 3)

    gdf.apply(Dataset_patches.check_pixel_nodata, axis=1, mask=mask)


def _filter_centroids(self):
    meta = self.raster.meta
    mask = self.raster.read_mask(1)
    patch_width, patch_height = self.patch_width, self.patch_height

    # set pixel boxes centroids as new geometry
    original_geometry = self.vector.geometry
    self.vector.set_geometry("box_pixel", inplace=True)

    # Crop the centroids to the raster bounds
    raster_pixel_bounds = shapely.box(
        patch_width // 2,
        patch_height // 2,
        meta["width"] - patch_width // 2,
        meta["height"] - patch_height // 2,
    )
    self.vector = self.vector.clip(raster_pixel_bounds)
    # set the original geometry back
    self.vector.set_geometry(original_geometry, inplace=True)

    # Check the nodata value for each patch,
    # the box_pixel (crop in pixel coordinates) is used as geometry
    records_mask = self.vector["box_pixel"].apply(self.check_pixel_nodata, mask=mask)
    self.vector = self.vector.loc[records_mask]

