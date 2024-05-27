import asyncio
import enum
import geopandas as gpd
import nest_asyncio
import numpy as np
import os
import osmnx as ox
import shapely

from blocksnet.preprocessing.utils import to_float
from blocksnet.utils import BUILDING_TYPES, TOTAL_FLATS, FLATS_PER_FLOOR, FLOORS_DEFAULTS, LIVING_BUILDING_TYPES, \
    CAPACITIES

# For the async work
nest_asyncio.apply()

LOCAL_CRS = 32637
CITY = "city"
AMENITY = "amenity"
BUILDING = "building"
METERS_PER_FLOOR = 2.7
AVERAGE_FLAT_AREA = 47.3
AVERAGE_AREA_PER_PERSON = 14


class FileFormatType(enum.Enum):
    """Class for specifying the format of output files."""
    GEOJSON = "geojson"
    PARQUET = "parquet"


class DataCollector:
    """Class for automatically loading input data of blocks generator"""

    __city: gpd.GeoDataFrame
    """GeoDataFrame of city"""
    __osm_id: str
    """osm_id of city in format R1234521 (`relation 1234521`)"""
    __file_path: os.path
    """Path to download data"""
    __output_file_format: FileFormatType
    """Format of output files"""
    __default_tags: list = [
        {"water": True},
        {"highway": [
            "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
            "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"
        ]},
        {"railway": ["tram", "rail"]},
    ]
    """Geometries required to blocks generator"""
    __allowable_geom_types: dict = {
        "water": ["Polygon", "LineString", "MultiPolygon"],
        "highway": ["LineString"],
        "railway": ["LineString"],
    }

    @property
    def __city_name(self) -> str:
        return str(self.__city["name"].iloc[0]).lower().replace(" ", "-")

    @property
    def __city_bbox(self) -> tuple:
        return self.__city.bbox_north, self.__city.bbox_south, self.__city.bbox_east, self.__city.bbox_west

    @property
    def __default_tag_names(self) -> set:
        return {tag_name for tag in self.__default_tags for tag_name, _ in tag.items()}

    def __init__(
            self,
            osm_id: str,
            output_file_format: FileFormatType = FileFormatType.GEOJSON,
            filepath: os.path = os.path.join("")
    ):
        self.__osm_id = osm_id
        self.__output_file_format = output_file_format
        self.__file_path = filepath
        self.__city = ox.geocode_to_gdf(self.__osm_id, by_osmid=True)

    def __filter_geometry(self, gdf: gpd.GeoDataFrame, tag: str) -> gpd.GeoDataFrame:
        gdf["geometry"] = gdf["geometry"].map(
            lambda x:
            x if self.__allowable_geom_types.get(tag) is None or x.geom_type in self.__allowable_geom_types[
                tag] else np.nan
        )
        return gdf.dropna(subset="geometry")

    @staticmethod
    def __geometry_to_point(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf["geometry"] = gdf["geometry"].map(lambda x: x if x.geom_type == "Point" else x.representative_point())
        return gdf

    def __file_name(self, param: str) -> str:
        return os.path.join(self.__file_path, f"{self.__city_name}_{param}.{self.__output_file_format.value}")

    def __to_json_file(self, gdf: gpd.GeoDataFrame, param: str, cols: list[str]) -> None:
        gdf.loc[:, cols].to_file(
            filename=self.__file_name(param),
            crs=LOCAL_CRS, driver="GeoJSON",
        )

    def __to_parquet_file(self, gdf: gpd.GeoDataFrame, param: str) -> None:
        gdf.to_parquet(
            path=self.__file_name(param),
        )

    def __to_file(self, gdf: gpd.GeoDataFrame, tag: str, cols: list[str]):
        if self.__output_file_format == FileFormatType.GEOJSON:
            self.__to_json_file(gdf, tag, cols)
        elif self.__output_file_format == FileFormatType.PARQUET:
            self.__to_parquet_file(gdf, tag)

    def __get_tag_name(self, tag: dict):
        tag_name = [
            tag_name if tag_name in self.__default_tag_names or type(tag_value) is bool else tag_value
            for tag_name, tag_value in tag.items()
        ][0]
        if type(tag_name) is list:
            tag_name = tag_name[0]
        return tag_name

    @staticmethod
    def __valid_building(cur_building: str) -> str:
        result = "yes"
        for building in BUILDING_TYPES:
            if cur_building.find(building) != -1:
                result = building
            break
        return result

    @staticmethod
    def __validate_buildings(gdf: gpd.GeoDataFrame):
        gdf["building"] = gdf["building"].apply(lambda x: DataCollector.__valid_building(x))

    @staticmethod
    def __valid_floors(raw):
        try:
            raw = str(raw)
            result = np.nan
            if raw.find("-") != -1:
                values = raw.split("-")
                result = int(float(values[1]) - float(values[0]) + 1)
            elif raw.find(";") != -1:
                result = int(raw.split(";")[0])
            elif raw.isdigit():
                result = int(raw)
            return result
        except:
            return np.nan

    @staticmethod
    def __validate_floors(gdf: gpd.GeoDataFrame):
        floors_is_not_na = ~gdf["floors"].isna()
        gdf.loc[floors_is_not_na, "floors"] = gdf[floors_is_not_na]["floors"].apply(
            lambda x: DataCollector.__valid_floors(x))
        gdf["floors"] = gdf["floors"].astype(float)

    @staticmethod
    def __validate_height(gdf):
        height_is_not_na = ~gdf["height"].isna()
        gdf.loc[height_is_not_na, "height"] = gdf[height_is_not_na]["height"].apply(lambda x: to_float(x))
        gdf["height"] = gdf["height"].astype(float)

    @staticmethod
    def __validate_flats(gdf):
        flats_is_not_na = ~gdf["flats"].isna()
        gdf.loc[flats_is_not_na, "flats"] = gdf[flats_is_not_na]["flats"].apply(lambda x: to_float(x))
        gdf["flats"] = gdf["flats"].astype(float)

    @staticmethod
    def __floors_to_height(gdf):
        height_is_na = gdf["height"].isna()
        floors_is_not_na = ~gdf["floors"].isna()

        gdf.loc[floors_is_not_na & height_is_na, "height"] = gdf[floors_is_not_na & height_is_na]["floors"].apply(
            lambda x: int(x) * METERS_PER_FLOOR
        )

    @staticmethod
    def __height_to_floors(gdf):
        floors_is_na = gdf["floors"].isna()
        height_is_not_na = ~gdf["height"].isna()

        gdf.loc[floors_is_na & height_is_not_na, "floors"] = gdf[floors_is_na & height_is_not_na]["height"].apply(
            lambda x: int(float(x) / METERS_PER_FLOOR)
        )

    @staticmethod
    def __fill_na_by_default_floors(gdf):
        floors_is_na = gdf["floors"].isna()
        gdf.loc[floors_is_na, "floors"] = gdf[floors_is_na]["building"].apply(lambda x: FLOORS_DEFAULTS.get(x, np.nan))

    @staticmethod
    def __total_flats_count(building):
        result = 0
        if building in TOTAL_FLATS.keys():
            result = TOTAL_FLATS[building]
        return result

    @staticmethod
    def __floors_to_flats(gdf):
        flats_is_na = gdf["flats"].isna()
        floors_is_not_na = ~gdf["floors"].isna()

        gdf.loc[flats_is_na & floors_is_not_na, "flats"] = gdf.loc[
            flats_is_na & floors_is_not_na, ["floors", "building"]].apply(
            lambda x: FLATS_PER_FLOOR[x["building"]] * int(x["floors"])
            if x["building"] in FLATS_PER_FLOOR.keys() else DataCollector.__total_flats_count(x["building"]), axis=1
        )

    @staticmethod
    def __fill_nan_gdf_params_with_median(gdf):
        flats_is_na = gdf["flats"].isna()
        floors_is_na = gdf["floors"].isna()
        height_is_na = gdf["height"].isna()
        flats_is_not_na = ~gdf["flats"].isna()
        floors_is_not_na = ~gdf["floors"].isna()
        height_is_not_na = ~gdf["height"].isna()

        for building in gdf["building"].unique():
            gdf.loc[(gdf["building"] == building) & flats_is_na, "flats"] = \
                gdf[(gdf["building"] == building) & flats_is_not_na]["flats"].median()
            gdf.loc[(gdf["building"] == building) & floors_is_na, "floors"] = \
                gdf[(gdf["building"] == building) & floors_is_not_na]["floors"].median()
            gdf.loc[(gdf["building"] == building) & height_is_na, "height"] = \
                gdf.loc[(gdf["building"] == building) & height_is_not_na, "height"].median()

    @staticmethod
    def __fill_is_living(gdf):
        gdf["is_living"] = gdf["building"].apply(
            lambda value: any([str(value).find(x) != -1 for x in LIVING_BUILDING_TYPES])
        )

    @staticmethod
    def __fill_living_area(gdf):
        gdf["living_area"] = gdf.apply(lambda x: int(x["flats"]) * AVERAGE_FLAT_AREA if x["is_living"] else 0, axis=1)

    @staticmethod
    def __fill_population(gdf):
        gdf["population"] = gdf.apply(
            lambda x: int(int(x["living_area"]) / AVERAGE_AREA_PER_PERSON) if x["is_living"] else 0, axis=1
        )

    @staticmethod
    def __fill_area(gdf):
        gdf["area"] = gdf["geometry"].apply(lambda x: x.area)

    def __download_city_boundaries(self) -> None:
        self.__city["geometry"] = self.__city["geometry"].to_crs(LOCAL_CRS)
        self.__to_file(self.__city, CITY, ["geometry"])

    async def __download_basic_geometry(self, tag: dict) -> None:
        tag_name = self.__get_tag_name(tag)
        try:
            gdf = ox.features_from_bbox(*self.__city_bbox, tags=tag)
            gdf["geometry"] = gdf["geometry"].to_crs(LOCAL_CRS)
            gdf = gpd.GeoDataFrame(geometry=gdf["geometry"]).reset_index(drop=True)
            gdf = self.__filter_geometry(gdf, tag_name)
            self.__to_file(gdf, tag_name, ["geometry"])
        except Exception as ex:
            print(f"Can't download geometry with name `{tag_name}`. Exception: ", ex)

    async def __download_geometry(self, tag: dict) -> None:
        tag_name = self.__get_tag_name(tag)
        try:
            gdf = ox.features_from_bbox(*self.__city_bbox, tags=tag)
            gdf["geometry"] = gdf["geometry"].to_crs(LOCAL_CRS)
            gdf["capacity"] = CAPACITIES.get(tag_name, 400)
            gdf = gpd.GeoDataFrame(gdf, geometry=gdf["geometry"]).reset_index(drop=True)
            gdf = self.__filter_geometry(gdf, tag_name)
            self.__geometry_to_point(gdf)
            self.__to_file(gdf, tag_name, ["geometry", "capacity"])
        except Exception as ex:
            print(f"Can't download service with name `{tag_name}`. Exception: ", ex)

    async def __download_buildings(self, tag: dict) -> None:
        try:
            gdf = ox.features_from_bbox(*self.__city_bbox, tags=tag)
            gdf["geometry"] = gdf["geometry"].to_crs(LOCAL_CRS)

            gdf = gdf.loc[:, ["geometry", "building", "building:levels", "height", "building:flats"]]
            gdf = gdf.rename(columns={"building:levels": "floors", "building:flats": "flats"})

            self.__validate_buildings(gdf)
            self.__validate_floors(gdf)
            self.__validate_height(gdf)
            self.__validate_flats(gdf)

            self.__floors_to_height(gdf)
            self.__height_to_floors(gdf)
            self.__fill_na_by_default_floors(gdf)
            self.__floors_to_flats(gdf)
            self.__fill_nan_gdf_params_with_median(gdf)

            self.__fill_is_living(gdf)
            self.__fill_living_area(gdf)
            self.__fill_population(gdf)
            self.__fill_area(gdf)
            self.__geometry_to_point(gdf)

            gdf = gdf.drop(columns=["height", "flats"])

            gdf = gdf[gdf["building"] != "yes"]

            gdf = gpd.GeoDataFrame(gdf, geometry=gdf["geometry"]).reset_index(drop=True)
            self.__to_file(gdf, "building", ["geometry", "population", "floors", "area", "living_area", "is_living"])
        except Exception as ex:
            print("Can't download buildings. Exception: ", ex)

    def __get_async_task(self, tag: dict):
        tag_name = [tag_name for tag_name, _ in tag.items()][0]
        if tag_name == AMENITY:
            return asyncio.create_task(self.__download_geometry(tag))
        elif tag_name == BUILDING:
            return asyncio.create_task(self.__download_buildings(tag))
        elif tag_name in self.__default_tag_names:
            return asyncio.create_task(self.__download_basic_geometry(tag))

    def __create_download_tasks(self, scenario: list[dict]) -> list:
        return [self.__get_async_task(elem) for elem in scenario]

    async def __collect(self, scenario: list[dict]):
        await asyncio.gather(*(self.__create_download_tasks(scenario)))

    async def __collect_basic_geometries(self):
        self.__download_city_boundaries()
        await asyncio.gather(*(self.__create_download_tasks(self.__default_tags)))

    def collect_basic_geometries(self) -> None:
        asyncio.run(self.__collect_basic_geometries())

    def collect(self, scenario: list[dict]) -> None:
        asyncio.run(self.__collect(scenario))
