import numpy as np
import rasterio
from typing import cast
import geopandas as gpd
from typing import Union, cast

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


def sample_raster(
    raster_path: str,
    vector_gdf: Union[gpd.GeoDataFrame, str],
    patch_width: int,
    patch_height: int,
    max_patches: int = 10
) -> gpd.GeoDataFrame:
    """
    Sample raster values for each geometry in the GeoDataFrame.

    Args:
        raster_path (str): Path to the raster file.
        vector_gdf (gpd.GeoDataFrame): GeoDataFrame with geometries.
        patch_width (int): Width of the patch to sample.
        patch_height (int): Height of the patch to sample.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with the sampled values.
    """
    if isinstance(vector_gdf, str):
        vector = gpd.read_file(vector_gdf)
        vector = cast(gpd.GeoDataFrame, vector)
    else:
        vector = vector_gdf.copy()

    # Open the raster file
    with rasterio.open(raster_path) as src:
        # read all bands:
        data = src.read()
        meta = src.meta
        # nodata value:
        nodatavals = src.nodatavals
#        bounds = src.bounds # format: (west, south, east, north)

    # transform from world coordinates to pixel coordinates
    pixel_to_world = src.meta["transform"]
    world_to_pixel = ~src.meta["transform"]

    # move the GeoDataFrame to the raster crs
    vector_ = vector.to_crs(meta["crs"])
    vector_ = cast(gpd.GeoDataFrame, vector_)

    # clip the geometries to the raster bounds
    centroid_px = vector_.geometry.centroid.affine_transform(
        world_to_pixel.to_shapely()
    )

    centroid_px = centroid_px.cx[
        0 : meta["width"], # x_min : x_max
        0 : meta["height"], # y_min : y_max
    ]

    # drop the centroids with no data
    for band in range(data.shape[0]):
        for i, row in enumerate(centroid_px):
            if row is None:
                continue
            x, y = int(row.x), int(row.y)
            if data[band, y, x] == nodatavals[band]:
                print(f"dropping centroid {i}")
                centroid_px.iloc[i] = None
    centroid_px = centroid_px.dropna()

    top_left_px = centroid_px.translate(
        xoff=-patch_width / 2, yoff=-patch_height / 2
    )
    bottom_right_px = centroid_px.translate(
        xoff=patch_width / 2, yoff=patch_height / 2
    )
    top_left_world = top_left_px.affine_transform(
        pixel_to_world.to_shapely()
    )
    bottom_right_world = bottom_right_px.affine_transform(
        pixel_to_world.to_shapely()
    )

    # get a patch around the centroids
    patch_meta = meta.copy()
    patch_meta["width"] = patch_width
    patch_meta["height"] = patch_height
    i = 0
    for tlw, brw, tlpx, brpx in zip(
        top_left_world, bottom_right_world, top_left_px, bottom_right_px
    ):
        x_min, y_min = int(tlpx.x), int(tlpx.y)
        x_max, y_max = int(brpx.x), int(brpx.y)
        patch = data[:, y_min:y_max, x_min:x_max]
        patch_meta["transform"] = rasterio.transform.from_bounds(
            tlw.x, brw.y, brw.x, tlw.y, patch_width, patch_height
        )
        with rasterio.open(f"patch_{i}_new_3.tif", "w", **patch_meta) as dst:
            for band in range(patch.shape[0]):
                dst.write(patch[band], band + 1)
        i += 1
        if i == max_patches:
            break

    





        


        

    return vector_gdf
```
