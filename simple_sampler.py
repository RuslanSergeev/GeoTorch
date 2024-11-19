""" This module contains the rasterio sampler class.
We encapsulate the rasterio sampler in a class to make it easier to use.
The sampler is able to extract raster values at the centroid of each polygon in a vector file.
It also samples the mask values effectivelly returning a flag if the reastered
patch contains nodata values.
"""

import rasterio
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import os
from typing import Union, List, Tuple, Optional, cast
from rasterio.transform import from_bounds


class RasterioSampler:

    DEFAULT_WIDTH = 64
    DEFAULT_HEIGHT = 64
    DEFAULT_OFFSET_X = 0
    DEFAULT_OFFSET_Y = 0

    def __init__(self, raster_path: str, vector_dataset: Union[str, gpd.GeoDataFrame]):
        """
        Initialize the RasterioSampler with the path to the raster file and a vector dataset.

        :param raster_path: Path to the raster file.
        :param vector_dataset: Path to the vector file or a GeoDataFrame.
        """
        self.raster_path = raster_path
        self.dataset = rasterio.open(raster_path)
        self.vector_gdf = self._open_vector(vector_dataset)
        self.vector_gdf["centroids"] = self.get_centroids()

    def __del__(self):
        """
        Close the raster dataset when the object is deleted.
        """
        self.dataset.close()

    def _open_vector(self, vector_dataset: Union[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        """
        Open the vector dataset and transform it to the raster's CRS if needed.

        :param vector_dataset: Path to the vector file or a GeoDataFrame.
        :return: GeoDataFrame transformed to the raster's CRS.
        """
        if isinstance(vector_dataset, str):
            gdf = gpd.read_file(vector_dataset)
        else:
            gdf = vector_dataset.copy()
        
        # Transform the CRS of the GeoDataFrame to match the raster's CRS
        if gdf.crs != self.dataset.crs:
            gdf.to_crs(self.dataset.crs, inplace=True)

        return gdf

    def get_centroids(self) -> gpd.GeoSeries:
        """
        In order to prevent computation of centroids at each sample retrieval, we compute them once.
        First estimate the utm crs of the vector dataset and then compute the centroids. 
        Move them back to the original crs.
        """
        self.vector_gdf = cast(gpd.GeoDataFrame, self.vector_gdf)
        return self.vector_gdf.geometry.to_crs(
            self.vector_gdf.estimate_utm_crs()
        ).centroid.to_crs(self.vector_gdf.crs)

    def get_row_col(self, 
        centroid: Point, 
        offset_x: int = DEFAULT_OFFSET_X,
        offset_y: int = DEFAULT_OFFSET_Y
    ) -> Union[Tuple[int, int], None]:
        """
        Get the row and column of the raster for a given centroid point, applying offsets.

        :param centroid: A Point representing the centroid of a feature.
        :param offset_x: Offset in pixels along the x-axis.
        :param offset_y: Offset in pixels along the y-axis.
        :return: Tuple of row, col coordinates or None if the centroid is out of bounds.
        """
        try:
            row, col = self.dataset.index(centroid.x, centroid.y)
        except ValueError:
            return None

        row += offset_y
        col += offset_x

        if row < 0 or col < 0 or row >= self.dataset.height or col >= self.dataset.width:
            return None

        return row, col

    def get_patch(self, 
        index: int,
        *,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        offset_x: int = DEFAULT_OFFSET_X,
        offset_y: int = DEFAULT_OFFSET_Y,
        bands: Optional[List[int]] = None, 
        filter_nodata: bool = True
    ) -> Union[np.ndarray, None]:
        """
        Extract raster values for a specific feature by index in the vector dataset.

        :param index: Index of the feature in the vector dataset.
        :param width: Width of the patch to extract.
        :param height: Height of the patch to extract.
        :param offset_x: Offset in pixels along the x-axis.
        :param offset_y: Offset in pixels along the y-axis.
        :param bands: List of band indices to extract.
        :return: An ndarray with the raster band values or np.nan if nodata is present.
        """
        centroid = self.vector_gdf["centroids"].iloc[index]

        row_col = self.get_row_col(centroid, offset_x, offset_y)
        if row_col is None:
            return None

        row, col = row_col

        if bands is None:
            bands = list(range(1, self.dataset.count + 1))
        
        band_values = []
        for band in bands:
            try:
                window = rasterio.windows.Window(col - width // 2, row - height // 2, width, height)
                band_data = self.dataset.read(band, window=window)
                mask_data = self.dataset.read_masks(band, window=window)
                if filter_nodata and np.any(mask_data == 0):
                    return None
                band_values.append(band_data)
            except IndexError:
                return None

        return np.array(band_values)

    def get_meta(self, 
        index: int, 
        *,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT, 
        offset_x: int = DEFAULT_OFFSET_X, 
        offset_y: int = DEFAULT_OFFSET_Y
    ) -> Union[dict, None]:
        """
        Get the metadata for a raster patch centered at a specific feature by index.

        :param index: Index of the feature in the vector dataset.
        :param width: Width of the patch to extract.
        :param height: Height of the patch to extract.
        :param offset_x: Offset in pixels along the x-axis.
        :param offset_y: Offset in pixels along the y-axis.
        :return: Metadata dictionary with updated transform component.
        """
        centroid = self.vector_gdf.geometry.centroid.iloc[index]
        if not isinstance(centroid, Point):
            return None

        row_col = self.get_row_col(centroid, offset_x, offset_y)
        if row_col is None:
            return None

        row, col = row_col

        window = rasterio.windows.Window(col - width // 2, row - height // 2, width, height)
        transform = from_bounds(*rasterio.windows.bounds(window, self.dataset.transform), width, height)
        meta = self.dataset.meta.copy()
        meta.update({
            'height': height,
            'width': width,
            'transform': transform
        })
        return meta

    def save_patch(self, 
        index: int,
        *,
        output_dir: str,
        filename_prefix: str,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        offset_x: int = DEFAULT_OFFSET_X,
        offset_y: int = DEFAULT_OFFSET_Y,
        bands: Optional[List[int]] = None
    ) -> None:
        """
        Sample the given raster for a specific feature and save the patch to the specified directory.

        :param index: Index of the feature in the vector dataset.
        :param output_dir: Directory where the patches will be saved.
        :param filename_prefix: Prefix for the output filename.
        :param width: Width of the patch to extract.
        :param height: Height of the patch to extract.
        :param offset_x: Offset in pixels along the x-axis.
        :param offset_y: Offset in pixels along the y-axis.
        :param bands: List of band indices to extract.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        patch = self.get_patch(
            index, width=width, height=height, offset_x=offset_x, offset_y=offset_y, bands=bands
        )
        meta = self.get_meta(
            index, width=width, height=height, offset_x=offset_x, offset_y=offset_y
        )

        if patch is None or meta is None:
            raise ValueError("Could not sample patch")

        if bands is None:
            bands = list(range(1, self.dataset.count + 1))

        if meta is not None:
            meta.update({'driver': 'GTiff', 'count': len(bands) if bands else self.dataset.count})
            output_path = os.path.join(output_dir, f"{filename_prefix}_{index}.tif")
            with rasterio.open(output_path, 'w', **meta) as dst:
                for j in range(len(bands)):
                    dst.write(patch[j], j + 1)

    def save(self,
        indexes: Optional[List[int]] = None,
        *,
        output_dir: str, 
        filename_prefix: str, 
        width: int = 64, 
        height: int = 64,
        offset_x: int = 0, 
        offset_y: int = 0, 
        bands: Union[List[int], None] = None
    ) -> None:
        """
        Sample the given raster for a list of features and save the patches to the specified directory.

        :param indexes: List of indexes of the features in the vector dataset.
        :param output_dir: Directory where the patches will be saved.
        :param filename_prefix: Prefix for the output filenames.
        :param width: Width of the patch to extract.
        :param height: Height of the patch to extract.
        :param offset_x: Offset in pixels along the x-axis.
        :param offset_y: Offset in pixels along the y-axis.
        :param bands: List of band indices to extract.
        """
        if indexes is None:
            indexes = list(range(len(self.vector_gdf)))
        for index in indexes:
            self.save_patch(
                index, output_dir=output_dir, filename_prefix=filename_prefix, width=width, height=height, offset_x=offset_x, offset_y=offset_y, bands=bands
            )


# Example usage
if __name__ == "__main__":
    raster_path = "path/to/your/raster.tif"
    vector_path = "path/to/your/vector.geojson"

    sampler = RasterioSampler(raster_path, vector_path)
    band_values = sampler.get_patch(0)
    patch_meta = sampler.get_meta(0)
    # Print results
    print(f"Band Values: {band_values}")
    print(f"Patch Metadata: {patch_meta}")

    # Save patch to output directory
    output_directory = "output_patches"
    sampler.save_patch(0, output_dir=output_directory, filename_prefix="patch")
