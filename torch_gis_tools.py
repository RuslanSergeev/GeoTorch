from typing import (
    Union, 
    List, 
    Tuple, 
    cast, 
    Dict, 
    Callable, 
    Optional
)
import geopandas as gpd
import numpy as np
import hashlib
import torch
import rasterio
from rasterstats import zonal_stats
from rasterio.features import rasterize


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
            padding=padding,
            bias=False
        )
        average_filter = self.create_circular_kernel(
            kernel_size, picture_resolution
        )
        conv_1.weight.data = average_filter
        conv_1.weight.requires_grad = False
        # Manually set the bias to zero and freeze it

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
                padding=padding,
                bias=False
            )
            # Manually set the kernel to an average filter: 1 / (kernel_size^2)
            average_filter = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
            conv_2.weight.data = average_filter
            conv_2.weight.requires_grad = False
            # Manually set the bias to zero and freeze it
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


def filter_bands(
    bands_names: List[str],
    bands_filter: List[str]
) -> List[str]:
    if not bands_filter:
        return bands_names
    filtered_bands = list(filter(lambda x: x in bands_names, bands_filter))
    if len(filtered_bands) != len(bands_filter):
        not_present_bands = set(bands_filter) - set(filtered_bands)
        raise ValueError(f"Bands {not_present_bands} not found in the raster")
    return filtered_bands


def filter_tensor_channels(
    tensor: torch.Tensor,
    bands_names: List[str],
    bands_filter: List[str] = []
) -> torch.Tensor:
    filtered_bands = filter_bands(bands_names, bands_filter)
    filtered_channels = [bands_names.index(band) for band in filtered_bands]
    return tensor[:, filtered_channels, :, :]


def read_raster_into_tensor(
    raster_path: str
) -> Tuple[torch.Tensor, List[str], dict]:
    with rasterio.open(raster_path) as src:
        meta = src.meta
        bands = list(src.descriptions)
        tensor = [torch.from_numpy(src.read(i)) for i in range(1, src.count + 1)]
        tensor = torch.stack(tensor, dim=0).unsqueeze(0).to(torch.float32)
    return tensor, bands, meta


def write_tensor_to_raster(
    tensor: torch.Tensor,
    raster_path: str,
    *,
    dtype: str = "float32",
    meta: dict,
    bands_names: List[str] = [],
) -> None:
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
        else:
            dst.descriptions = [None] * tensor.shape[1]


def estimate_density(
    raster_input_path: str,
    raster_output_path: str,
    density_buffer: int,
    smoothing_buffer: int = 0,
    bands_filter: List[str] = []
):
    # read the raster file
    tensor, bands_names, meta = read_raster_into_tensor(raster_input_path)

    if bands_filter:
        tensor = filter_tensor_channels(
            tensor, bands_names, bands_filter
        )

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
        density,
        raster_output_path,
        meta=meta,
        bands_names=bands_names
    )


def rasterize_geodataframe(
    gdf: gpd.GeoDataFrame,
    *,
    pixel_size_metters: int = 4,
    raster_path: str,
    features_list: List[str] = [],
    rasterize_centroids: bool = True,
    centroid_value: int = 1,
    centroid_name: str = "centroid_",
    fill: float = 0
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
    transform = rasterio.transform.from_origin( #type: ignore
        west = bounds[0], 
        north = bounds[3],
        xsize = pixel_size_metters, 
        ysize = pixel_size_metters
    )
    width = int((bounds[2] - bounds[0]) / pixel_size_metters)
    height = int((bounds[3] - bounds[1]) / pixel_size_metters)

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
            shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[feature])]
            out_image = rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=fill, #type: ignore
            )
            dst.write(out_image, band)
        # describe the bands
        dst.descriptions = features_list


# compute zonal stats of a raster 
# using a geo-dataframe as zones
def get_zonal_statistics(
    raster_path: str,
    gdf_input: Union[gpd.GeoDataFrame, str],
    *,
    stats: List[str] = ["mean"],
    add_stats: Dict[str, Callable] = {},
    band: int = 1,
    nodata: Optional[float] = None,
    gdf_output_path: Optional[str] = None
) -> gpd.GeoDataFrame:
    # read the raster file
    if isinstance(gdf_input, str):
        gdf_input = gpd.read_file(gdf_input)
    else:
        gdf_input = gdf_input.copy()
    gdf_input = cast(gpd.GeoDataFrame, gdf_input)
    origin_crs = gdf_input.crs
    with rasterio.open(raster_path) as src:
        meta = src.meta
        gdf_input.to_crs(meta["crs"], inplace=True)
        stats_out = zonal_stats(
            gdf_input, 
            src.read(band), 
            stats=stats,
            band=band,
            add_stats=add_stats,
            affine=src.transform,
            nodata=nodata
        )
    stats_out = gpd.GeoDataFrame(
        data=stats_out, 
        geometry = gdf_input.geometry
    ).to_crs(origin_crs)
    stats_out = cast(gpd.GeoDataFrame, stats_out)
    if gdf_output_path:
        stats_out.to_file(gdf_output_path, driver="GPKG")
    return stats_out

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
