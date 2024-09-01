import collections
from .accessibility import Accessibility
from .base_method import BaseMethod
from blocksnet.models.land_use import LandUse
from blocksnet.method.provision import Provision
from blocksnet.models import Block
import pandas as pd
import numpy as np
from typing import Tuple
from pulp import *
from blocksnet.method.provision import Provision
import matplotlib.pyplot as plt
from collections import defaultdict


class Indicator:
    def __init__(self, FSI_min, FSI_max, site_coverage_min, site_coverage_max):
        self.FSI_min = FSI_min  # минимальный коэффициент плотности застройки
        self.FSI_max = FSI_max  # максимальный коэффициент плотности застройки
        self.site_coverage_min = site_coverage_min  # минимальный процент застроенности участка
        self.site_coverage_max = site_coverage_max  # максимальный процент застроенности участка


LAND_USE_INDICATORS = {
    LandUse.RESIDENTIAL: Indicator(FSI_min=0.5, FSI_max=3.0, site_coverage_min=0.2, site_coverage_max=0.8),
    LandUse.BUSINESS: Indicator(FSI_min=1.0, FSI_max=3.0, site_coverage_min=0.0, site_coverage_max=0.8),
    LandUse.RECREATION: Indicator(FSI_min=0.05, FSI_max=0.2, site_coverage_min=0.0, site_coverage_max=0.3),
    LandUse.SPECIAL: Indicator(FSI_min=0.05, FSI_max=0.2, site_coverage_min=0.05, site_coverage_max=0.15),
    LandUse.INDUSTRIAL: Indicator(FSI_min=0.3, FSI_max=1.5, site_coverage_min=0.2, site_coverage_max=0.8),
    LandUse.AGRICULTURE: Indicator(FSI_min=0.1, FSI_max=0.2, site_coverage_min=0.0, site_coverage_max=0.6),
    LandUse.TRANSPORT: Indicator(FSI_min=0.2, FSI_max=1.0, site_coverage_min=0.0, site_coverage_max=0.8),
}


class BlockOptimizer(BaseMethod):
    BLOCKS_LANUSE_DICT: dict = {}
    ORIG_SERVICE_TYPES: dict = {}
    NEW_SERVICE_TYPES: dict = {}
    FREE_AREA: dict = {}
    BRICKS: dict = {}
    CONSTANTS: dict = {}
    SCENARIO: dict = {}
    FREE_AREA: Tuple = (0, 0)
    MAX_FACILITIES: int = None

    def calculate_free_area(self, selected_block_id):
        selected_block = self.city_model[selected_block_id]
        land_use_coef = 0.8
        return selected_block.site_area * land_use_coef

    def get_capacity_demand(self, selected_block_id):
        gdf = self.city_model.get_blocks_gdf(False)

        selected_block_gdf = gdf.loc[selected_block_id].to_dict()
        gdf.loc[selected_block_id, "population"] = 0

        selected_block = self.city_model[selected_block_id]
        acc_gdf = Accessibility(city_model=self.city_model).calculate(selected_block)

        constants = {}
        min_capacity_left = float("inf")
        min_service = None

        for serv in list(
            set(self.NEW_SERVICE_TYPES[selected_block_id]) | set(self.ORIG_SERVICE_TYPES[selected_block_id])
        ):
            idxs = acc_gdf[
                (acc_gdf.accessibility_to <= serv.accessibility) | (acc_gdf.accessibility_from <= serv.accessibility)
            ]["id"]
            capacity_column = f"capacity_{serv.name}"
            context_gdf = gdf.loc[idxs]

            demand_local = context_gdf["population"].apply(serv.calculate_in_need).sum()
            capacity_local = context_gdf[capacity_column].sum() - selected_block_gdf[capacity_column]

            demand_global = gdf["population"].apply(serv.calculate_in_need).sum()
            capacity_global = gdf[capacity_column].sum() - selected_block_gdf[capacity_column]

            capacity_left = capacity_global - demand_global
            if capacity_left < min_capacity_left:
                min_capacity_left = capacity_left
                min_service = serv

            constants[serv.name] = {"demand_local": demand_local, "capacity_local": capacity_local}

        min_population = max(min_capacity_left / min_service.demand * 1000, 0)

        return constants, min_population

    def get_bricks_df(self, service_types):
        bricks_dict_list = []

        for serv in service_types:
            for brick in serv.bricks:
                brick_dict = {
                    "service_type": serv.name,
                    "capacity": brick.capacity,
                    "area": brick.area,
                    # "is_integrated": brick.is_integrated,
                }
                bricks_dict_list.append(brick_dict)
        bricks_df = pd.DataFrame(bricks_dict_list)
        return bricks_df

    def get_optimal_update_df(self, prob):
        n_bricks = len(prob.variables())
        counts = {}
        for key in self.BRICKS:
            counts[key] = np.zeros(len(self.BRICKS[key]))
        populations = {}

        service_counts = prob.variables()

        for var_idx in range(n_bricks):

            var = service_counts[var_idx]
            service_count = var.value()
            block_idx, idx = var.name.rsplit("_")

            if idx == "population":
                populations[int(block_idx)] = service_count
            else:
                block_idx, idx = int(block_idx), int(idx)
                counts[block_idx][idx] = service_count
                self.BRICKS[block_idx].loc[idx, "capacity_agg"] = (
                    self.BRICKS[block_idx].loc[idx, "capacity"] * service_count
                )
                self.BRICKS[block_idx].loc[idx, "area_agg"] = self.BRICKS[block_idx].loc[idx, "area"] * service_count

        update = {}
        bricks_to_build_df = {}
        for key in self.BLOCKS_LANUSE_DICT:
            bricks_to_build = np.where(counts[key] != 0.0)[0]
            bricks_to_build_df[key] = self.BRICKS[key].loc[bricks_to_build]
            service_counts = counts[key][counts[key] != 0.0]
            bricks_to_build_df[key]["service_counts"] = service_counts
            constants, min_population = self.CONSTANTS[key]
            bricks_to_build_df[key]["demand_in_need"] = bricks_to_build_df[key]["service_type"].apply(
                lambda x: constants[x]["demand_local"] - constants[x]["capacity_local"]
            )

            service_capacities = bricks_to_build_df[key].groupby("service_type").sum()["capacity_agg"].to_dict()

            population = populations.get(key, 0)
            if population > 0:
                service_capacities["population"] = population

            update[key] = service_capacities

        update_df = pd.DataFrame.from_dict(update, orient="index")
        return update_df, bricks_to_build_df, prob.objective.value()

    def generate_lp_problem(self):
        prob = LpProblem("ProvisionOpt", LpMaximize)

        lp_sum_components = []

        for key in self.BLOCKS_LANUSE_DICT:
            if len(self.SCENARIO) != 0:
                servs = [serv.name for serv in self.NEW_SERVICE_TYPES[key]]
                intersecting_keys = set(self.SCENARIO.keys()).intersection(servs)
                scenario_coef_sum = sum(self.SCENARIO[key] for key in intersecting_keys)
            else:
                scenario_coef_sum = 0

            default_weight = (1 - scenario_coef_sum) / len(self.NEW_SERVICE_TYPES[key])
            bricks_df = self.BRICKS[key]
            constants, min_population = self.CONSTANTS[key]
            service_counts = LpVariable.dicts(f"{key}", list(bricks_df.index), 0, self.MAX_FACILITIES, cat=LpInteger)
            population = LpVariable(f"{key}_population", 0, None, cat=LpInteger)

            for serv in self.NEW_SERVICE_TYPES[key]:
                service_bricks_idxs = list(bricks_df[bricks_df.service_type == serv.name].index)

                demand_local, capacity_local = (
                    constants[serv.name]["demand_local"],
                    constants[serv.name]["capacity_local"],
                )

                fit_function = lpSum(service_counts[n] * bricks_df.loc[n].capacity for n in service_bricks_idxs)

                demand_from_new_population = (population * serv.demand) / 1000
                C_i = fit_function + capacity_local
                D_i = demand_local + demand_from_new_population
                service_weight = self.SCENARIO.get(serv.name, default_weight)

                if demand_local > 0 and capacity_local < demand_local:
                    lp_sum_components.append(service_weight * (C_i - D_i))
                    prob += fit_function <= demand_local - capacity_local

                if capacity_local > demand_local and self.BLOCKS_LANUSE_DICT[key] == LandUse.RESIDENTIAL:
                    prob += demand_from_new_population <= capacity_local - demand_local

            block_area = self.city_model[key].site_area

            indicators = LAND_USE_INDICATORS[self.BLOCKS_LANUSE_DICT[key]]

            FSI_max, FSI_min = indicators.FSI_max, indicators.FSI_min

            prob += (
                sum(service_counts[n] * bricks_df.loc[n].area / block_area for n in list(bricks_df.index)) <= FSI_max
            )
            prob += (
                sum(service_counts[n] * bricks_df.loc[n].area / block_area for n in list(bricks_df.index)) >= FSI_min
            )

            prob += sum(service_counts[n] * bricks_df.loc[n].area for n in list(bricks_df.index)) <= block_area

        prob += sum(i for i in lp_sum_components)

        return prob

    def get_deleting_update_df(self, selected_block_id, delete_population: bool = True):

        service_capacities = {}
        selected_block = self.city_model[selected_block_id]
        for serv in selected_block.all_services:
            service_dict = serv.to_dict()
            service_capacities[service_dict["service_type"]] = -service_dict["capacity"]

        if delete_population:
            service_capacities["population"] = -selected_block.population

        update = {selected_block_id: service_capacities}

        update_df = pd.DataFrame.from_dict(update, orient="index")
        return update_df

    def plot(
        self,
        optimal_update_df,
        bricks_to_build_df,
        total_before,
        total_after,
        categories,
        previous_values,
        current_values,
    ):
        differences = np.round(np.array(current_values) - np.array(previous_values), 2)

        filtered_categories = []
        filtered_differences = []
        filtered_previous_values = []

        highlight_columns = list(optimal_update_df.columns)

        for category, diff, prev in zip(categories, differences, previous_values):
            if diff != 0:
                filtered_categories.append(category)
                filtered_differences.append(diff)
                filtered_previous_values.append(prev)

        previous_values = np.array(filtered_previous_values)
        categories = np.array(filtered_categories)
        differences = np.array(filtered_differences)

        bar_width = 0.5
        num_bars = len(filtered_categories)

        fig_width = num_bars * bar_width * 1.5

        fig_width = fig_width if fig_width > 10 else 10
        fig_height = 6

        if len(categories) == 0:
            plt.text(
                0.5,
                0.5,
                "Изменений нет",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=14,
                color="red",
            )
            plt.axis("off")
        else:
            become_higher = differences > 0
            become_lower = differences < 0

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            bars_previous = ax.bar(categories, previous_values, width=bar_width, label="Previous Values", color="blue")

            bars_difference_higher = ax.bar(
                categories[become_higher],
                differences[become_higher],
                width=bar_width,
                bottom=previous_values[become_higher],
                color="green",
                label="Provision become higher",
            )

            for bar, diff, prev, category in zip(
                bars_difference_higher,
                differences[become_higher],
                previous_values[become_higher],
                categories[become_higher],
            ):
                yval = bar.get_height() if diff > 0 else 0
                text = ax.text(bar.get_x() + bar.get_width() / 2, prev + yval + 0.01, f"{round(diff, 2)}", ha="center")

            bars_difference_lower = ax.bar(
                categories[become_lower],
                differences[become_lower],
                width=bar_width,
                bottom=previous_values[become_lower],
                color="red",
                label="Provision become lower",
            )

            for bar, diff, prev, category in zip(
                bars_difference_lower,
                differences[become_lower],
                previous_values[become_lower],
                categories[become_lower],
            ):
                yval = bar.get_height() if diff > 0 else 0
                text = ax.text(bar.get_x() + bar.get_width() / 2, prev + yval + 0.01, f"{round(diff, 2)}", ha="center")

            ax.set_ylim(0.0, 1.1)

            land_use_names = f"Changing {self.ORIG_LANDUSE.name.capitalize()} to {self.NEW_LANDUSE.name.capitalize()}"
            total_provision_difference = f"Total provision difference: {round(total_after - total_before, 3)}"

            ax.set_xlabel("City Services\n The services highlighted in orange have just been built")
            ax.set_ylabel("Provision")
            ax.set_title(
                f"Differences of Provisions in city services\n {land_use_names} \n {total_provision_difference}"
            )
            ax.legend()

            plt.xticks(rotation=45)

            for i, tick in enumerate(plt.gca().get_xticklabels()):
                if tick.get_text() in highlight_columns:
                    tick.set_color("orange")

            plt.tight_layout()

            plt.show()

    def get_provision_diff(self, optimal_update_df):
        prov = Provision(city_model=self.city_model)

        orig_landuse_services = self.city_model.get_land_use_service_types(self.ORIG_LANDUSE)
        new_landuse_services = self.city_model.get_land_use_service_types(self.NEW_LANDUSE)

        orig_landuse_services_set = set([x.name for x in orig_landuse_services])
        new_landuse_services_set = set([x.name for x in new_landuse_services])

        combined_landuse_services = list(orig_landuse_services_set | new_landuse_services_set)

        scenario = {elem: 1 / len(combined_landuse_services) for elem in combined_landuse_services}

        gdf, total_before = prov.calculate_scenario(scenario, self_supply=True)
        provision_before = [prov.total_provision(gdf[service]) for service in scenario]

        deleting_update_df = self.get_deleting_update_df(selected_block_id=self.SELECTED_BLOCK, delete_population=True)

        update_df = optimal_update_df.combine_first(deleting_update_df)

        gdf, total_after = prov.calculate_scenario(scenario, update_df=update_df, self_supply=True)

        provision_after = [prov.total_provision(gdf[service]) for service in scenario]

        return total_before, total_after, combined_landuse_services, provision_before, provision_after

    def calculate(self, blocks_landuse_dict: dict, scenario: dict = {}, max_facilities: int = None):
        self.SCENARIO = scenario
        self.BLOCKS_LANUSE_DICT = blocks_landuse_dict
        self.MAX_FACILITIES = max_facilities

        self.ORIG_SERVICE_TYPES = {}
        self.NEW_SERVICE_TYPES = {}
        self.FREE_AREA = {}
        self.BRICKS = {}
        self.CONSTANTS = {}
        for key in self.BLOCKS_LANUSE_DICT:
            orig_landuse = self.city_model[key].land_use
            orig_service_types = self.city_model.get_land_use_service_types(orig_landuse)
            new_service_types = self.city_model.get_land_use_service_types(blocks_landuse_dict[key])
            self.ORIG_SERVICE_TYPES[key] = orig_service_types
            self.NEW_SERVICE_TYPES[key] = new_service_types
            self.FREE_AREA[key] = self.calculate_free_area(key)
            self.CONSTANTS[key] = self.get_capacity_demand(key)
            self.BRICKS[key] = self.get_bricks_df(new_service_types)

        # method
        prob = self.generate_lp_problem()
        prob.solve(PULP_CBC_CMD(msg=False))
        print(LpStatus[prob.status])
        optimal_update_df, bricks_to_build_df, provision = self.get_optimal_update_df(prob)

        dfs = []
        for key in self.BLOCKS_LANUSE_DICT:
            deleting_update_df = self.get_deleting_update_df(selected_block_id=key, delete_population=True)
            dfs.append(deleting_update_df)
        combined_df = pd.concat(dfs, ignore_index=True)

        return {
            "optimal_update_df": optimal_update_df.fillna(0),
            "bricks_to_build_df": bricks_to_build_df,
            "deleting_df": combined_df.fillna(0),
            "provision": provision,
        }
