import rasterio
from typing import Union, Tuple, Dict, Any, Optional
import geopandas as gpd
import numpy as np
import shapely


class Dataset_patches:
    def __init__(
        self,
        vector: Union[gpd.GeoDataFrame, str],
        raster: Union[rasterio.DatasetReader, str],
        patch_width: int,
        patch_height: int
    ):
        self.patch_width, self.patch_height = patch_width, patch_height
        self._open_sources(vector, raster)
        self._convert_crs()
        self._get_patches_boxes()
        self._filter_centroids()
        self._add_centroid_identifier()
        self._create_patch_meta_template()


    def _convert_crs(self):
        if self.vector.crs != self.raster.crs:
            self.vector.to_crs(self.raster.crs, inplace=True)

    
    def _open_sources(
        self,
        vector: Union[gpd.GeoDataFrame, str],
        raster: Union[rasterio.DatasetReader, str],
    ):
        if isinstance(vector, str):
            self.vector = gpd.read_file(vector)
        else:
            self.vector = vector.copy()
        if isinstance(raster, str):
            self.raster = rasterio.open(raster)
        else:
            self.raster = raster


    def _add_centroid_identifier(self):
        # Calculate centroid for each building polygon
        centroids = self._get_centroids()

        # id = x _ y with x and y being the coordinates of the centroid
        # enough precision for 50 cm accuracy
        self.vector["object_id"] = centroids.x.astype(str) + "_" + centroids.y.astype(str)


    def _get_centroids(self):
        original_crs = self.vector.crs
        utm_crs = self.vector.geometry.estimate_utm_crs()
        utm_centroids = self.vector.geometry.to_crs(utm_crs).centroid
        return utm_centroids.to_crs(original_crs)


    def _get_patches_boxes(
        self
    ):
        self.transform_px_to_world = self.raster.meta["transform"]
        self.transform_world_to_px = ~self.transform_px_to_world

        # disable the warning about the crs
        centroid_world = self._get_centroids()
        centroid_px = centroid_world.affine_transform(
            self.transform_world_to_px.to_shapely()
        )
        # Define the box corners in pixels coordinates.
        top_left_px = centroid_px.translate(
            xoff=-self.patch_width / 2, yoff=-self.patch_height / 2
        )
        bottom_right_px = centroid_px.translate(
            xoff=self.patch_width / 2, yoff=self.patch_height / 2
        )
        # Define the box corners in world coordinates.
        top_left_world = top_left_px.affine_transform(
            self.transform_px_to_world.to_shapely()
        )
        bottom_right_world = bottom_right_px.affine_transform(
            self.transform_px_to_world.to_shapely()
        )
        # Add patches in world and pixel coordinates to 
        # the dataframe member
        self.vector["box_pixel"] = [
            shapely.box(tl.x, tl.y, br.x, br.y)
            for tl, br in zip(top_left_px, bottom_right_px)
        ]
        self.vector["box_world"] = [
            shapely.box(tl.x, tl.y, br.x, br.y)
            for tl, br in zip(top_left_world, bottom_right_world)
        ]


    @staticmethod
    def check_pixel_nodata(pixel_geometry, mask):
        # if geometry is a point, check if the point is nodata
        if pixel_geometry.geom_type == "Point":
            return mask[pixel_geometry.y, pixel_geometry.x]
        else:
            # check top left corner:
            min_x, min_y, max_x, max_y = map(int, pixel_geometry.bounds)
            return np.all(mask[min_y:max_y, min_x:max_x])


    def _filter_centroids(self):
        meta = self.raster.meta
        mask = self.raster.read_masks(1)
        patch_width, patch_height = self.patch_width, self.patch_height

        # set pixel boxes centroids as new geometry
        self.vector.set_geometry("box_pixel", inplace=True)

        # Crop the centroids to the raster bounds
        # Add a margin of 1 pixel to avoid the edge of the raster
        self.vector = self.vector.cx[
            patch_width // 2+1 : meta["width"] - patch_width // 2 - 1,
            patch_height // 2+1 : meta["height"] - patch_height // 2 - 1,
        ]

        # Check the nodata value for each patch,
        # the box_pixel (crop in pixel coordinates) is used as geometry
        records_mask = self.vector.geometry.apply(self.check_pixel_nodata, mask=mask)
        self.vector = self.vector.loc[records_mask]

        # set the original geometry back
        self.vector.set_geometry("geometry", inplace=True)


    def _create_patch_meta_template(self):
        self.meta_template = self.raster.meta.copy()
        self.meta_template["width"] = self.patch_width
        self.meta_template["height"] = self.patch_height


    def sample_patch(
        self, 
        index: int, 
        out_path: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if index >= len(self.vector) or index < 0:
            raise IndexError("Index out of bounds")

        # Get the pixel coordinates of the box
        x_min, y_min, x_max, y_max = map(int, self.vector.iloc[index].box_pixel.bounds)
        # Read the patch from the raster
        patch = self.raster.read(window=((y_min, y_max), (x_min, x_max)))
        patch_meta = self.meta_template.copy()
        patch_meta["transform"] = rasterio.transform.from_bounds(
            *self.vector.iloc[index].box_world.bounds,
            self.patch_width,
            self.patch_height
        )
        
        if out_path:
            with rasterio.open(out_path, "w", **patch_meta) as dst:
                for band in range(patch.shape[0]):
                    dst.write(patch[band], band + 1)
        return patch, patch_meta


    def save_patches(self, out_path_prefix: str):
        for i in range(len(self.vector)):
            object_id = self.vector.iloc[i]["object_id"]
            self.sample_patch(i, out_path_prefix + f"_{object_id}.tif")


    def save_vector(self, out_path: str):
        self.vector.to_file(out_path)


    def __len__(self):
        return len(self.vector)

    
    def __getitem__(self, index):
        return self.sample_patch(index, None)


# def sample_raster(
#     raster_data: Union[rasterio.DatasetReader, str],
#     vector_gdf: Union[gpd.GeoDataFrame, str],
#     patch_width: int,
#     patch_height: int,
#     max_patches: int = 10
# ) -> gpd.GeoDataFrame:
#     """
#     Sample raster values for each geometry in the GeoDataFrame.
# 
#     Args:
#         raster_path (str): Path to the raster file.
#         vector_gdf (gpd.GeoDataFrame): GeoDataFrame with geometries.
#         patch_width (int): Width of the patch to sample.
#         patch_height (int): Height of the patch to sample.
# 
#     Returns:
#         gpd.GeoDataFrame: GeoDataFrame with the sampled values.
#     """
# 
#     # Filter centroids
#     centroid_px = filter_centroids(
#         gdf_boxes=boxes,
#         patch_width=patch_width,
#         patch_height=patch_height,
#         data=data,
#         meta=meta,
#     )
# 
#     # get a patch around the centroids
#     patch_meta = meta.copy()
#     patch_meta["width"] = patch_width
#     patch_meta["height"] = patch_height
#     i = 0
#     for _, row in centroid_px.iterrows():
#         # the bbox is west, south, east, north
#         west_px, south_px, east_px, north_px = row.box_pixel.bounds
#         west_w, south_w, east_w, north_w = row.box_world.bounds
#         patch = data[:, north_px:south_px, west_px:east_px]
#         patch_meta["transform"] = rasterio.transform.from_bounds(
#             west_w, north_w, east_w, south_w, patch_width, patch_height
#         )
#         with rasterio.open(f"patch_{i}_new_3.tif", "w", **patch_meta) as dst:
#             for band in range(patch.shape[0]):
#                 dst.write(patch[band], band + 1)
#         i += 1
#         if i == max_patches:
#             break
