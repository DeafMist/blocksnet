
import osmnx as ox
import pandas as pd
import geopandas as gpd
from typing import ClassVar

from pydantic import BaseModel
from ...models import CityModel


class VacantArea(BaseModel):

    city_model: CityModel
 
    local_crs: ClassVar[int] = 32636
    roads_buffer: ClassVar[int] = 10
    buildings_buffer: ClassVar[int] = 10

    @staticmethod
    def dwn_other(block, local_crs) -> gpd.GeoSeries:
        '''download other'''
        try:
            other = ox.features_from_polygon(block, tags={"man_made": True, "aeroway": True,"military": True })
            other['geometry'] = other['geometry'].to_crs(local_crs)
            return other.geometry
        except:
            return gpd.GeoSeries()
    
    @staticmethod
    def dwn_leisure(block, local_crs) -> gpd.GeoSeries:
        try:
            leisure = ox.features_from_polygon(block, tags={"leisure": True})
            leisure['geometry'] = leisure['geometry'].to_crs(local_crs)
            return leisure.geometry
        except:
            return gpd.GeoSeries()
    
    @staticmethod
    def dwn_landuse(block, local_crs) -> gpd.GeoSeries:
        try:
            landuse = ox.features_from_polygon(block, tags={"landuse": True})
            if not landuse.empty:
                landuse = landuse[landuse['landuse'] != 'residential']
            landuse['geometry'] = landuse['geometry'].to_crs(local_crs)
            return landuse.geometry
        except:
            return gpd.GeoSeries()
    
    @staticmethod
    def dwn_amenity(block, local_crs) -> gpd.GeoSeries:
        try:
            amenity = ox.features_from_polygon(block, tags={"amenity": True})
            amenity['geometry'] = amenity['geometry'].to_crs(local_crs)
            return amenity.geometry
        except:
            return gpd.GeoSeries()

    @staticmethod
    def dwn_buildings(block, local_crs ,buildings_buffer) -> gpd.GeoSeries:
        try:
            buildings = ox.features_from_polygon(block, tags={"building": True})
            if buildings_buffer:
                buildings['geometry'] = buildings['geometry'].to_crs(local_crs).buffer(buildings_buffer)
            return buildings.geometry
        except:
            return gpd.GeoSeries()
    
    @staticmethod
    def dwn_natural(block, local_crs) -> gpd.GeoSeries:
        try:
            natural = ox.features_from_polygon(block, tags={"natural": True})
            if not natural.empty:
                natural = natural[natural['natural'] != 'bay']
            natural['geometry'] = natural['geometry'].to_crs(local_crs)
            return natural.geometry
        except:
            return gpd.GeoSeries()
    
    @staticmethod
    def dwn_waterway(block, local_crs) -> gpd.GeoSeries:
        try:
            waterway = ox.features_from_polygon(block, tags={"waterway": True})
            waterway['geometry'] = waterway['geometry'].to_crs(local_crs)
            return waterway.geometry
        except:
            return gpd.GeoSeries()
    
    @staticmethod
    def dwn_highway(block, local_crs, roads_buffer) -> gpd.GeoSeries:
        try:
            highway = ox.features_from_polygon(block, tags={"highway": True})
            condition = (highway['highway'] != 'path') & (highway['highway'] != 'footway') & (highway['highway'] != 'pedestrian')
            filtered_highway = highway[condition]
            if roads_buffer:
                filtered_highway['geometry'] = filtered_highway['geometry'].to_crs(local_crs).buffer(roads_buffer)
            filtered_highway['geometry'] = filtered_highway['geometry'].to_crs(local_crs).buffer(1)
            return filtered_highway.geometry
        except:
            return gpd.GeoSeries()
    
    @staticmethod
    def dwn_path(block, local_crs)-> gpd.GeoSeries:
        try:
            tags = {'highway': 'path', 'highway': 'footway'}
            path = ox.features_from_polygon(block, tags=tags)
            path['geometry'] = path['geometry'].to_crs(local_crs).buffer(0.5)
            return path.geometry
        except:
            return gpd.GeoSeries()
    
    @staticmethod
    def dwn_railway(block, local_crs) -> gpd.GeoSeries:
        try:
            railway = ox.features_from_polygon(block, tags={"railway": True})
            if not railway.empty:
                railway = railway[railway['railway'] != 'subway']
            railway['geometry'] = railway['geometry'].to_crs(local_crs)
            return railway.geometry
        except:
            return gpd.GeoSeries()
    
    @staticmethod
    def create_minimum_bounding_rectangle(polygon) -> gpd.GeoSeries:
        return polygon.minimum_rotated_rectangle
    
    @staticmethod
    def buffer_and_union(row, buffer_distance=1) -> gpd.GeoSeries:
        polygon = row['geometry']
        buffer_polygon = polygon.buffer(buffer_distance)
        return buffer_polygon
        
    def get_vacant_area(self, blpck_id:int) -> gpd.GeoDataFrame:
        
        blocks = self.city_model.blocks.to_gdf().copy()
        blocks = gpd.GeoDataFrame(geometry=gpd.GeoSeries(blocks.geometry))
        if blpck_id:
            block_gdf = gpd.GeoDataFrame([blocks.iloc[blpck_id]], crs=blocks.crs)
            block_buffer = block_gdf['geometry'].buffer(20).to_crs(epsg=4326).iloc[0]
        else:
            block_gdf = blocks
            block_buffer = blocks.buffer(20).to_crs(epsg=4326).unary_union

        leisure = self.dwn_leisure(block_buffer, self.local_crs)
        landuse = self.dwn_landuse(block_buffer, self.local_crs)
        other  = self.dwn_other(block_buffer, self.local_crs)
        amenity = self.dwn_amenity(block_buffer, self.local_crs)
        buildings = self.dwn_buildings(block_buffer, self.local_crs, self.buildings_buffer)
        natural = self.dwn_natural(block_buffer, self.local_crs)
        waterway = self.dwn_waterway(block_buffer, self.local_crs)
        highway = self.dwn_highway(block_buffer, self.local_crs, self.roads_buffer)
        railway = self.dwn_railway(block_buffer, self.local_crs)
        path = self.dwn_path(block_buffer, self.local_crs)

        occupied_area = [leisure, other, landuse, amenity, buildings, natural, waterway, highway, railway, path]
        occupied_area = pd.concat(occupied_area)
        occupied_area = gpd.GeoDataFrame(geometry=gpd.GeoSeries(occupied_area))

        block_buffer2 = gpd.GeoDataFrame(geometry=block_gdf.buffer(60))
        polygon = occupied_area.geometry.geom_type == "Polygon"
        multipolygon = occupied_area.geometry.geom_type == "MultiPolygon"
        blocks_new = gpd.overlay(block_buffer2, occupied_area[polygon], how="difference")
        blocks_new = gpd.overlay(blocks_new, occupied_area[multipolygon], how="difference")
        blocks_new = gpd.overlay(block_gdf, blocks_new, how="intersection")
        blocks_exploded = blocks_new.explode(index_parts=True)
        blocks_exploded.reset_index(drop=True,inplace=True)

        blocks_filtered_area = blocks_exploded[blocks_exploded['geometry'].area >= 100] #1
        if blocks_filtered_area.empty:
            print('The block has no vacant area')
            return  None
        area_attitude = 1.9 # 2
        for index, row in blocks_filtered_area.iterrows():
            polygon = row['geometry']
            mbr = self.create_minimum_bounding_rectangle(polygon)
            if polygon.area * area_attitude < mbr.area:
                blocks_filtered_area.drop(index, inplace=True)

        blocks_filtered_area['buffered_geometry'] = blocks_filtered_area.apply(self.buffer_and_union, axis=1)
        unified_geometry = blocks_filtered_area['buffered_geometry'].unary_union

        result_gdf = gpd.GeoDataFrame(geometry=[unified_geometry], crs=blocks_filtered_area.crs)
        result_gdf_exploded = result_gdf.explode(index_parts=True)
        result_gdf_exploded['area'] = result_gdf_exploded['geometry'].area
        result_gdf_exploded.reset_index(drop=True, inplace=True)
        return result_gdf_exploded
