from typing import Union, List
import geopandas as gpd
import pandas as pd
import numpy as np
import hashlib
import torch
import rasterio
from rasterio.features import rasterize

PATH_GRID = "Noto_grid.gpkg"
PATH_BUILDINGS = "CORRECTED_japan_noto_bfgt_20240801.gpkg"
PATH_LOW_DENSITY = "Noto_BF_low_density.gpkg"


class DensityEstimator(torch.nn.Module):
    def __init__(self,
        *,
        density_buffer: int, 
        smoothing_buffer: int, 
        picture_resolution: int,
    ):
        super().__init__()
        self.net = torch.nn.Sequential()

        # Set up the density estimator network
        kernel_size = int(density_buffer / picture_resolution)
        kernel_size += 1 - (kernel_size % 2)
        padding = (kernel_size - 1) // 2

        conv_1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding
        )
        average_filter = self.create_circular_kernel(kernel_size, picture_resolution)
        conv_1.weight.data = average_filter
        conv_1.weight.requires_grad = False
        # Manually set the bias to zero and freeze it
        conv_1.bias.data = torch.zeros(1)
        conv_1.bias.requires_grad = False

        self.net.add_module("conv_1", conv_1)

        # If smoothing buffer provided, add a second convolutional layer
        # to smooth the density map
        if smoothing_buffer > 0:
            kernel_size = int(smoothing_buffer / picture_resolution)
            kernel_size += 1 - (kernel_size % 2)
            padding = (kernel_size - 1) // 2
            conv_2 = torch.nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                padding=padding
            )
            # Manually set the kernel to an average filter: 1 / (kernel_size^2)
            average_filter = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
            conv_2.weight.data = average_filter
            conv_2.weight.requires_grad = False
            # Manually set the bias to zero and freeze it
            conv_2.bias.data = torch.zeros(1)
            conv_2.bias.requires_grad = False
            self.net.add_module("conv_2", conv_2)

    def forward(self, x):
        N, C, H, W = x.shape
        reshaped = x.view(N * C, 1, H, W)
        return self.net(reshaped).view(N, C, H, W)

    @staticmethod
    def create_circular_kernel(
        kernel_size_px: int, 
        picture_resolution: int
    ) -> torch.Tensor:
        diameter_size_px = kernel_size_px
        radius_px = diameter_size_px / 2
        radius_m = radius_px * picture_resolution
        area_m2 = np.pi * (radius_m ** 2)
        kernel = np.zeros((diameter_size_px, diameter_size_px))
        for i in range(diameter_size_px):
            for j in range(diameter_size_px):
                x_m = i*picture_resolution - radius_m
                y_m = j*picture_resolution - radius_m
                if (x_m ** 2 + y_m ** 2) <= (radius_m ** 2):
                    kernel[i, j] = 1
                else:
                    kernel[i, j] = 0
        return torch.from_numpy(kernel / area_m2)[None, None, :, :].to(torch.float32)


def read_raster_into_tensor(
    raster_path: str,
) -> torch.Tensor:
    with rasterio.open(raster_path) as src:
        meta = src.meta
        bands_names = src.descriptions
        tensor = [torch.from_numpy(src.read(i)) for i in range(1, src.count + 1)]
        tensor = torch.stack(tensor, dim=0).unsqueeze(0).to(torch.float32)
    return tensor, bands_names, meta


def write_tensor_to_raster(
    raster_path: str,
    tensor: torch.Tensor,
    *,
    dtype: str = "float32",
    meta: dict,
    bands_names: List[str] = [],
):
    # switch datatype to float8
    if bands_names and len(bands_names) != tensor.shape[1]:
        raise ValueError("Bands names should be the same as number of channels")
    meta["dtype"] = dtype
    with rasterio.open(raster_path, 'w', **meta) as dst:
        for channel in range(tensor.shape[1]):
            dst.write(
                tensor[0, channel, :, :].numpy(force=True), channel + 1
            )
        if bands_names:
            dst.descriptions = bands_names


def estimate_density(
    raster_input_path: str,
    raster_output_path: str,
    density_buffer: int,
    smoothing_buffer: int = 0
):
    # read the raster file
    tensor, bands_names, meta = read_raster_into_tensor(raster_input_path)

    # create the density estimator
    density_estimator = DensityEstimator(
        density_buffer=density_buffer, 
        smoothing_buffer=smoothing_buffer, 
        picture_resolution=meta["transform"].a
    )

    if torch.cuda.is_available():
        density_estimator = density_estimator.cuda()
        tensor = tensor.cuda()

    # estimate the density
    density = density_estimator(tensor)

    # write the density to a raster file
    write_tensor_to_raster(
        raster_output_path,
        density,
        meta=meta,
        bands_names=bands_names
    )


def write_centroids_to_raster(
    gdf: gpd.GeoDataFrame,
    *,
    cell_size: int = 4,
    raster_path: str,
    features_list: List[str] = [],
    rasterize_centroids: bool = True,
    centroid_value: int = 1,
    centroid_name: str = "centroid_",
):
    if isinstance(gdf, str):
        gdf = gpd.read_file(gdf)

    gdf = gdf.copy()
    utm_crs = gdf.estimate_utm_crs()
    gdf.to_crs(utm_crs, inplace=True)
    gdf.set_geometry(gdf.centroid, inplace=True)
    if rasterize_centroids:
        gdf[centroid_name] = centroid_value
        features_list.append(centroid_name)

    bounds = gdf.total_bounds
    transform = rasterio.transform.from_origin(
        west = bounds[0], 
        north = bounds[3],
        xsize = cell_size, 
        ysize = cell_size
    )
    width = int((bounds[2] - bounds[0]) / cell_size)
    height = int((bounds[3] - bounds[1]) / cell_size)

    out_meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(features_list),
        "dtype": "float32",
        "crs": utm_crs,
        "transform": transform
    }

    with rasterio.open(raster_path, 'w', **out_meta) as dst:
        for band, feature in enumerate(features_list, start=1):
            out_image = rasterize(
                [(geom, value) for geom, value in zip(gdf.geometry, gdf[feature])],
                out_shape=(height, width),
                transform=transform,
                fill=0
            )
            dst.write(out_image, band)
        # describe the bands
        dst.descriptions = features_list


# compute zonal stats of a raster 
# using a geo-dataframe as zones
def compute_zonal_statistics(
    raster_path: str,
    gdf: Union[gpd.GeoDataFrame, str],
    *,
    stats: List[str] = ["mean"],
) -> gpd.GeoDataFrame:
    # read the raster file
    with rasterio.open(raster_path) as src:
        raster = src.read(1)
        meta = src.meta
    if isinstance(gdf, str):
        gdf = gpd.read_file(gdf)
    gdf = gdf.copy()
    gdf.to_crs(meta["crs"], inplace=True)

    # compute zonal statistics
    stats = rasterio.features.zonal_stats(
        gdf.geometry,
        raster,
        affine=meta["transform"],
        stats=stats
    )
    stats = pd.DataFrame(stats, index=gdf.index)
    gdf = pd.concat([gdf, stats], axis=1)
    return gdf



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


def get_density_within_buffer(
    gdf_objects: gpd.GeoDataFrame,
    buffer_radius: float,
    id_column_name: str = "object_id",
    density_column_name: str = "density"
) -> gpd.GeoDataFrame:

    utm_crs = gdf_objects.estimate_utm_crs()

    # prepare the hash IDs collection
    gdf_buffers = gdf_objects.copy()
    gdf_buffers.to_crs(utm_crs, inplace=True)
    gdf_buffers = create_centroid_identifier(
        gdf_buffers,
        id_column_name=id_column_name
    )

    gdf_ids = gdf_buffers.copy()
    gdf_ids.to_crs(utm_crs, inplace=True)
    gdf_ids.geometry = gdf_ids.centroid

    # create a buffer around each building
    gdf_buffers["geometry"] = gdf_buffers.buffer(buffer_radius)

    # count the number of objects in each buffer
    sjoin = gpd.sjoin(
        gdf_ids, gdf_buffers, how='inner', predicate='within',
    )
    sizes = sjoin.groupby(id_column_name).count()
    sizes[id_column_name] = sizes.index.copy()
    sizes[density_column_name] = sizes["geometry"]
    sjoin = sjoin.merge(
        sizes[[id_column_name, density_column_name]], on=id_column_name
    )

    gdf_ids = gdf_ids.merge(
        sjoin[[id_column_name, density_column_name]], on=id_column_name
    )
    return gdf_ids


def get_density(
    gdf_objects: gpd.GeoDataFrame,
    gdf_cells: gpd.GeoDataFrame,
    id_column_name: str = "object_id",
    density_column_name: str = "density"
) -> gpd.GeoDataFrame:
    # prepare the hash IDs collection
    gdf_ids = gdf_objects.copy()
    gdf_ids = create_centroid_identifier(
        gdf_ids,
        id_column_name=id_column_name
    )

    gdf_cells = gdf_cells.copy()
    gdf_cells["patch_id"] = gdf_cells.index.copy()

    # simplify the geometries to centroids
    geometry_backup = gdf_ids.geometry.copy()
    gdf_ids = gdf_ids.set_geometry(gdf_ids.centroid)

    # count the number of objects in each cell
    sjoin = gpd.sjoin(
        gdf_ids, gdf_cells, how='right', predicate='within'
    )
    sizes = sjoin.groupby("patch_id").count()
    sizes["patch_id"] = sizes.index.copy()
    sizes[density_column_name] = sizes[id_column_name]
    sjoin = sjoin.merge(
        sizes[["patch_id", density_column_name]], on="patch_id"
    )

    gdf_ids = gdf_ids.merge(
        sjoin[[id_column_name, density_column_name]], on=id_column_name
    )
    gdf_ids = gdf_ids.set_geometry(geometry_backup)
    return gdf_ids


def filter_density(
    gdf_objects: gpd.GeoDataFrame,
    gdf_cells: gpd.GeoDataFrame,
    num_min: int = 5,
    num_max: int = 15
) -> gpd.GeoDataFrame:
    # prepare the hash IDs collection
    gdf_ids = gdf_objects.copy()
    gdf_ids = create_centroid_identifier(gdf_ids)

    # simplify the geometries to centroids
    geometry_backup = gdf_ids.geometry.copy()
    gdf_ids = gdf_ids.set_geometry(gdf_ids.centroid)

    # filter
    right_join = gpd.sjoin(gdf_ids, gdf_cells, how='right', predicate='within')
    sizes = right_join.groupby("id").size()
    sizes = sizes.loc[(sizes >= num_min) & (sizes < num_max)]
    right_join = right_join.loc[right_join["id"].isin(sizes.index)]
    gdf_ids = gdf_ids.loc[gdf_ids.building_id.isin(right_join.building_id)]
    gdf_ids = gdf_ids.set_geometry(geometry_backup)
    return gdf_ids


gdf_cells = gpd.read_file(PATH_GRID)
gdf_buildings = gpd.read_file(PATH_BUILDINGS)
gdf_low_density = filter_density(gdf_buildings, gdf_cells)



