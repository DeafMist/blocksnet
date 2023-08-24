"""
Class holding parameters for land use filter parameters is defined here.
"""
import geopandas as gpd
from pydantic import BaseModel, Field, field_validator
from masterplan_tools.models.geojson import PolygonGeoJSON


class LandUseFeatureProperties(BaseModel):
    id: int | None = None


class LandUseParameters(BaseModel):
    """
    Parameters used for land use filter
    """

    landuse: PolygonGeoJSON[LandUseFeatureProperties]
    """basic landuse geometries"""
    no_development: PolygonGeoJSON[LandUseFeatureProperties]
    """territories with restricted development"""
    buildings: PolygonGeoJSON[LandUseFeatureProperties] = None
    """buildings geometries that are used for clustering inside of blocks"""

    @field_validator("landuse", "no_development", "buildings", mode="before")
    def validate_fields(value):
        if isinstance(value, gpd.GeoDataFrame):
            return PolygonGeoJSON[LandUseFeatureProperties].from_gdf(value)
        return value
