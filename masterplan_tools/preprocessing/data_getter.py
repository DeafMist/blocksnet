"""
This module gets data from the OSM. The module also implements several methods of data processing.
These methods allow you to connect parts of the data processing pipeline.
"""

import geopandas as gpd
import networkx as nx
import pandas as pd
from tqdm.auto import tqdm
from typing import Literal
from pydantic import BaseModel, field_validator

from .aggregate_parameters import AggregateParameters
from ..models.city_model import CityBlockFeature, AccessibilityMatrix
from ..models import PolygonGeoJSON
from ..method.blocks.blocks_cutter import BlocksCutterFeatureProperties
from .accs_matrix_calculator import Accessibility

tqdm.pandas()


class DataGetter(BaseModel):
    """
    This class is used to get and pre-process data to be used in calculations in other modules.
    """

    blocks: PolygonGeoJSON[BlocksCutterFeatureProperties]

    @field_validator("blocks", mode="before")
    def validate_blocks(value):
        if isinstance(value, gpd.GeoDataFrame):
            return PolygonGeoJSON[BlocksCutterFeatureProperties].from_gdf(value)
        return value

    def get_accessibility_matrix(self, graph: nx.Graph) -> AccessibilityMatrix:
        """
        This function returns an accessibility matrix for a city. The matrix is calculated using
        the `Accessibility` class.

        Args:
            blocks (GeoDataFrame, optional): A GeoDataFrame containing information about the blocks in the city.
            Defaults to None.
            graph (Graph, optional): A networkx graph representing the city's road network. Defaults to None.

        Returns:
            np.ndarray: An accessibility matrix for the city.
        """

        accessibility = Accessibility(self.blocks.to_gdf(), graph)
        return AccessibilityMatrix(df=accessibility.get_matrix())

    @staticmethod
    def _get_living_area(row) -> float:
        """
        This function calculates the living area of a building based on the data in the given row.

        Args:
            row (pd.Series): A row of data containing information about a building.

        Returns:
            float: The calculated living area of the building.
        """

        if row["living_area"]:
            return float(row["living_area"])
        if row["is_living"]:
            if row["storeys_count"]:
                if row["building_area"]:
                    living_area = row["building_area"] * row["storeys_count"] * 0.7

                    return living_area
                return 0.0
            return 0.0
        return 0.0

    @staticmethod
    def _get_living_area_pyatno(row) -> float:
        """
        This function calculates the living area of a building based on the data in the given row.
        If the `living_area` attribute is not available, the function returns 0.

        Args:
            row (pd.Series): A row of data containing information about a building.

        Returns:
            float: The calculated living area of the building.
        """

        if row["living_area"]:
            return float(row["building_area"])
        return 0.0

    def aggregate_blocks_info(self, params: AggregateParameters) -> "PolygonGeoJSON[CityBlockFeature]":
        """
        This function aggregates information about blocks in a city. The information includes data about buildings,
        green spaces, and parking spaces.

        Args:
            blocks (gpd.GeoDataFrame): A GeoDataFrame containing information about the blocks in the city.
            buildings (gpd.GeoDataFrame): A GeoDataFrame containing information about buildings in the city.
            greenings (gpd.GeoDataFrame): A GeoDataFrame containing information about green spaces in the city.
            parkings (gpd.GeoDataFrame): A GeoDataFrame containing information about parking spaces in the city.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing aggregated information about blocks in the city.
        """

        blocks = self.blocks.to_gdf()
        buildings = params.buildings.to_gdf()
        greenings = params.greenings.to_gdf()
        parkings = params.parkings.to_gdf()

        buildings["living_area"].fillna(0, inplace=True)
        buildings["storeys_count"].fillna(0, inplace=True)
        tqdm.pandas(desc="Restoring living area")
        # TODO: is tqdm really necessary?
        buildings["living_area"] = buildings.progress_apply(self._get_living_area, axis=1)
        tqdm.pandas(desc="Restoring living area squash")
        # TODO: is tqdm really necessary?
        buildings["living_area_pyatno"] = buildings.progress_apply(self._get_living_area_pyatno, axis=1)
        buildings["total_area"] = buildings["building_area"] * buildings["storeys_count"]

        blocks_and_greens = (
            gpd.sjoin(blocks, greenings, predicate="intersects", how="left")
            .groupby("id")
            .agg(
                {
                    "current_green_capacity": "sum",
                    "current_green_area": "sum",
                }
            )
        )
        blocks_and_greens = (
            blocks_and_greens.reset_index(drop=True).reset_index(drop=False).rename(columns={"index": "block_id"})
        )

        blocks_and_parkings = (
            gpd.sjoin(blocks, parkings, predicate="intersects", how="left")
            .groupby("id")
            .agg({"current_parking_capacity": "sum"})
        )
        blocks_and_parkings = (
            blocks_and_parkings.reset_index(drop=True).reset_index(drop=False).rename(columns={"index": "block_id"})
        )

        blocks_and_buildings = (
            gpd.sjoin(blocks, buildings, predicate="intersects", how="left")
            .drop(columns=["index_right"])
            .groupby("id")
            .agg(
                {
                    "population_balanced": "sum",
                    "building_area": "sum",
                    "storeys_count": "median",
                    "total_area": "sum",
                    "living_area": "sum",
                    "living_area_pyatno": "sum",
                }
            )
        )
        blocks_and_buildings = (
            blocks_and_buildings.reset_index(drop=True).reset_index(drop=False).rename(columns={"index": "block_id"})
        )

        blocks.reset_index(drop=False, inplace=True)

        blocks_info_aggregated = pd.merge(blocks_and_buildings, blocks_and_greens)
        blocks_info_aggregated = pd.merge(blocks_info_aggregated, blocks_and_parkings)

        blocks_info_aggregated = gpd.GeoDataFrame(
            pd.merge(blocks, blocks_info_aggregated, left_on="index", right_on="block_id").drop(
                columns=["index", "id"]
            ),
            geometry="geometry",
        )
        blocks_info_aggregated.rename(
            columns={"building_area": "building_area_pyatno", "total_area": "building_area"}, inplace=True
        )

        blocks_info_aggregated["current_industrial_area"] = (
            blocks_info_aggregated["building_area_pyatno"] - blocks_info_aggregated["living_area_pyatno"]
        )
        blocks_info_aggregated.rename(
            columns={
                "population_balanced": "current_population",
                "storeys_count": "floors",
                "living_area_pyatno": "current_living_area",
            },
            inplace=True,
        )
        blocks_info_aggregated["area"] = blocks_info_aggregated["geometry"].area
        blocks_info_aggregated.drop(columns=["building_area_pyatno", "building_area", "living_area"], inplace=True)

        return PolygonGeoJSON[CityBlockFeature].from_gdf(blocks_info_aggregated)

    @classmethod
    def prepare_graph(  # pylint: disable=too-many-arguments,too-many-locals
        cls,
        blocks,
        service_type=None,
        service_gdf=None,
        accessibility_matrix=None,
        buildings=None,
        updated_block_info=None,
        services_graph=None,
    ):
        """
        This function prepares a graph for calculating the provision of a specified service in a city.

        Args:
            blocks (gpd.GeoDataFrame): A GeoDataFrame containing information about the blocks in the city.
            service_type (str, optional): The type of service to calculate the provision for. Defaults to None.
            service_gdf (gpd.GeoDataFrame, optional): A GeoDataFrame containing information about blocks with the
            specified service in the city. Defaults to None.
            accessibility_matrix (np.ndarray, optional): An accessibility matrix for the city. Defaults to None.
            buildings (gpd.GeoDataFrame, optional): A GeoDataFrame containing information about buildings in the city.
            Defaults to None.
            updated_block_info (gpd.GeoDataFrame, optional): A GeoDataFrame containing updated information
            about blocks in the city. Defaults to None.

        Returns:
            nx.Graph: A networkx graph representing the city's road network with additional data for calculating
            the provision of the specified service.
        """
        service = service_gdf

        blocks_with_buildings = (
            gpd.sjoin(blocks, buildings, predicate="intersects", how="left")
            .drop(columns=["index_right"])
            .groupby("id")
            .agg({"population_balanced": "sum"})
        )

        blocks_with_buildings.reset_index(drop=False, inplace=True)
        blocks_with_buildings["is_living"] = blocks_with_buildings["population_balanced"].apply(lambda x: x > 0)

        blocks_crs = blocks.crs.to_epsg()

        blocks = blocks_with_buildings.merge(blocks, right_on="id", left_on="id")
        blocks = gpd.GeoDataFrame(blocks, geometry="geometry", crs=blocks_crs)

        living_blocks = blocks.loc[:, ["id", "geometry"]].sort_values(by="id").reset_index(drop=True)

        service_gdf = (
            gpd.sjoin(blocks, service, predicate="intersects")
            .groupby("id")
            .agg(
                {
                    "capacity": "sum",
                }
            )
        )

        if updated_block_info:
            for updated_block in updated_block_info.values():
                if updated_block["block_id"] not in service_gdf.index:
                    service_gdf.loc[updated_block["block_id"]] = 0
                if service_type == "recreational_areas":
                    service_gdf.loc[updated_block["block_id"], "capacity"] += updated_block.get("G_max_capacity", 0)
                else:
                    service_gdf.loc[updated_block["block_id"], "capacity"] += updated_block.get(
                        f"{service_type}_capacity", 0
                    )

            blocks.loc[updated_block["block_id"], "population_balanced"] = updated_block["population"]

        blocks_geom_dict = blocks[["id", "population_balanced", "is_living"]].set_index("id").to_dict()
        service_blocks_dict = service_gdf.to_dict()["capacity"]

        blocks_list = accessibility_matrix.loc[
            accessibility_matrix.index.isin(service_gdf.index.astype("Int64")),
            accessibility_matrix.columns.isin(living_blocks["id"]),
        ]

        # TODO: is tqdm really necessary?
        for idx in list(blocks_list.index):
            blocks_list_tmp = blocks_list[blocks_list.index == idx]
            blocks_list.columns = blocks_list.columns.astype(int)
            blocks_list_tmp_dict = blocks_list_tmp.transpose().to_dict()[idx]

            for key in blocks_list_tmp_dict.keys():
                if key != idx:
                    services_graph.add_edge(idx, key, weight=round(blocks_list_tmp_dict[key], 1))

                else:
                    services_graph.add_node(idx)

                services_graph.nodes[key]["population"] = blocks_geom_dict["population_balanced"][int(key)]
                services_graph.nodes[key]["is_living"] = blocks_geom_dict["is_living"][int(key)]

                if key != idx:
                    try:
                        if services_graph.nodes[key][f"is_{service_type}_service"] != 1:
                            services_graph.nodes[key][f"is_{service_type}_service"] = 0
                            services_graph.nodes[key][f"provision_{service_type}"] = 0
                            services_graph.nodes[key][f"id_{service_type}"] = 0
                    except KeyError:
                        services_graph.nodes[key][f"is_{service_type}_service"] = 0
                        services_graph.nodes[key][f"provision_{service_type}"] = 0
                        services_graph.nodes[key][f"id_{service_type}"] = 0
                else:
                    services_graph.nodes[key][f"is_{service_type}_service"] = 1
                    services_graph.nodes[key][f"{service_type}_capacity"] = service_blocks_dict[key]
                    services_graph.nodes[key][f"provision_{service_type}"] = 0
                    services_graph.nodes[key][f"id_{service_type}"] = 0

                if services_graph.nodes[key]["is_living"]:
                    services_graph.nodes[key][f"population_prov_{service_type}"] = 0
                    services_graph.nodes[key][f"population_unprov_{service_type}"] = blocks_geom_dict[
                        "population_balanced"
                    ][int(key)]

        return services_graph
