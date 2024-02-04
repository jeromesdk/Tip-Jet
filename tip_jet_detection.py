"""
Created on 31/01/2024
@author: Hind FARIS and Jérôme SIOC'HAN DE KERSABIEC

Class with useful functions to analyse Tip-Jet events.
"""

from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

# gcsfs to read data from online storage
import gcsfs

# sklearn for the EOF decomposition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# mit to find consecutive numbers in a given list
import more_itertools as mit

# pyproj to compute clean distance between longitudes and latitudes
from pyproj import Geod

# to transform indices into real date from an xarray
import netCDF4

g = Geod(ellps='WGS84')


def consecutive_groups(
        indices: np.ndarray
) -> list[list]:
    """
    Function to detect consecutive events related to a same Tip-Jet event

    :param indices: A list containing time indices of Tip-Jet events
    :return: A list of list containing time indices of consecutive Tip-Jet events
    """
    consec = []
    for group in mit.consecutive_groups(indices):
        consec.append(list(group))
    return np.array(consec, dtype='object')


class Member:
    """
    A class with tools to detect Tip-Jet events on a member of CMIP6 datasets (for future scenarios).
    """

    def __init__(
            self,
            experiment_id: str,
            source_id: str,
            member_id: str,
            activity_id: str = 'ScenarioMIP',
            table_id: str = 'day',
            threshold: float = 3.11534
    ):
        """
        Initialises a member by setting its attributes

        :param experiment_id: type of forcing used for the runs, "historical" is based on past observations
        :param source_id: the model used
        :param member_id: id of a member
        :param activity_id: type of CMIP experiment. Historical run are in the "CMIP" while future scenarios are in
                            "ScenarioMIP"
        :param table_id: frequency of data spanning
        :param threshold: The threshold value to use on PCA to determine whether it is a Tip-Jet
        """
        self.threshold = threshold
        self.activity_id = activity_id
        self.table_id = table_id
        self.experiment_id: str = experiment_id
        self.source_id = source_id
        self.member_id = member_id
        self.df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
        self.ds_uas, self.ds_uas_area = self.zonal_wind_speed()
        self.ds_windmag, self.wind_mag_area, self.argmax_wind, self.u_at_max = self.sea_surface_wind_speed()
        self.temp_area, self.t_at_max = self.sea_surface_temperature()
        self.grad_a, self.grad_b, self.grad_c = self.sea_surface_pressure()
        self.eof = self.eof()

    def zonal_wind_speed(self) -> Tuple[xr.core.dataarray.DataArray, xr.core.dataarray.DataArray]:
        """
        Extracts zonal wind speed on CMIP6 dataset in the world and in the area of interest (south greenland)

        :return: zonal wind speed on the globe and the area of interest
        """
        df_uas = self.df.query(
            f"activity_id=='{self.activity_id}' & table_id=='{self.table_id}' & experiment_id=='{self.experiment_id}' & variable_id=='uas' & source_id=='{self.source_id}' & member_id=='{self.member_id}'")
        gcs = gcsfs.GCSFileSystem(token='anon')
        zstore = df_uas.zstore.values[-1]
        mapper = gcs.get_mapper(zstore)
        ds_uas = xr.open_zarr(mapper, consolidated=True, decode_times=False).uas

        # We select the area concerned by the Tip-Jet (greenland south)
        ds_uas_area = ds_uas.where(
            (ds_uas.lat > 57) & (ds_uas.lat < 61) & (ds_uas.lon > 360 - 46) & (
                    ds_uas.lon < 360 - 33), drop=True)
        return ds_uas, ds_uas_area

    def sea_surface_wind_speed(self) \
            -> Tuple[xr.core.dataarray.DataArray, xr.core.dataarray.DataArray, np.ndarray, np.ndarray]:
        """
        Extracts sea surface wind speed on CMIP6 dataset in the world and in the area of interest (south greenland).
        Gives an array with the indices (coordinates) of the point with the maximum wind speed for each day. For those
        indices gives the zonal wind speed.

        :return: wind magnitude around the world and in the area of interest, the indices (coordinates) of the point
        with the maximum wind speed for each day. For those points give the zonal wind speed.
        """
        df_sfcWind = self.df.query(
            f"activity_id=='{self.activity_id}' & table_id=='{self.table_id}' & experiment_id=='{self.experiment_id}' "
            f"& variable_id=='sfcWind' & source_id=='{self.source_id}' & member_id=='{self.member_id}'")
        gcs = gcsfs.GCSFileSystem(token='anon')
        zstore = df_sfcWind.zstore.values[-1]
        mapper = gcs.get_mapper(zstore)
        ds_windmag = xr.open_zarr(mapper, consolidated=True, decode_times=False).sfcWind

        # We select the area concerned by the Tip-Jet (greenland south)
        ds_windmag = ds_windmag.isel(time=np.arange(self.ds_uas_area.time.size))
        wind_mag_area = ds_windmag.where(
            (ds_windmag.lat > 57) & (ds_windmag.lat < 61) & (ds_windmag.lon > 360 - 46) & (
                    ds_windmag.lon < 360 - 33), drop=True)

        # we convert the xr.DataArray to np.array to make the processing easier
        wind_magnitude = wind_mag_area.values

        # we retrieve the daily point having the maximal wind (both zonal (wind from west to east) and meridional
        # (wind from south to north)) 'sfcWind'
        argmax_wind = np.argmax(np.array([wind_magnitude[i].ravel() for i in range(len(wind_magnitude))]), axis=1)

        # We retrieve uas at the same point as identified before having maximal wind
        u_np = self.ds_uas_area.values
        u_at_max = np.array([u_np[i].ravel()[argmax_wind[i]] for i in range(len(u_np))])
        return ds_windmag, wind_mag_area, argmax_wind, u_at_max

    def sea_surface_temperature(self) -> Tuple[xr.core.dataarray.DataArray, np.ndarray]:
        """
        Gives daily sea surface temperature (SST) in the area of interest (south greenland). Also gives the daily SST at
        the coordinates where the max speed of wind is detected.

        :return: Tuple of daily SST in the area of interest and daily SST at max speed coordinates
        """
        df_tas = self.df.query(
            f"activity_id=='{self.activity_id}' & table_id=='{self.table_id}' & experiment_id=='{self.experiment_id}' "
            f"& variable_id=='tas' & source_id=='{self.source_id}' & member_id=='{self.member_id}'")
        gcs = gcsfs.GCSFileSystem(token='anon')
        zstore = df_tas.zstore.values[-1]
        mapper = gcs.get_mapper(zstore)
        ds_temp = xr.open_zarr(mapper, consolidated=True, decode_times=False).tas

        # We select the area concerned by the Tip-Jet (greenland south)
        temp_area = ds_temp.where(
            (ds_temp.lat > 57) & (ds_temp.lat < 61) & (ds_temp.lon > 360 - 46) & (ds_temp.lon < 360 - 33), drop=True)

        # We retrieve temperature  at the same point as identified before having maximal wind
        u_np = temp_area.values
        t_at_max = np.array([u_np[i].ravel()[self.argmax_wind[i]] for i in range(len(u_np))]) - 273.15
        return temp_area, t_at_max

    def sea_surface_pressure(self) -> Tuple[xr.core.dataarray.DataArray, xr.core.dataarray.DataArray, xr.core.dataarray.DataArray]:
        """
        Gives 3 daily sea gradient pressures between 3 points to another point. Those points are chose to be maximised
        during Tip-Jet events.

        :return: 3 daily sea gradient pressures data
        """
        df_psl = self.df.query(
            f"activity_id=='{self.activity_id}' & table_id=='{self.table_id}' & experiment_id=='{self.experiment_id}' & variable_id=='psl' & source_id=='{self.source_id}' & member_id=='{self.member_id}'")
        gcs = gcsfs.GCSFileSystem(token='anon')
        zstore = df_psl.zstore.values[-1]
        mapper = gcs.get_mapper(zstore)
        ds_slp = xr.open_zarr(mapper, consolidated=True, decode_times=False).psl

        # We select the area concerned by the Tip-Jet (greenland south)
        ds_slp_area = ds_slp.where(
            (ds_slp.lat > 57) & (ds_slp.lat < 61) & (ds_slp.lon > 360 - 46) & (ds_slp.lon < 360 - 33), drop=True)

        # We select the nearest Lon/Lat to the points defined in the paper
        lon_a, lat_a = 360 + np.linspace(-47, -40, 20), np.linspace(60, 62, 20)
        lon_b, lat_b = 360 + np.linspace(-45, -40, 20), np.linspace(57, 62, 20)
        lon_c, lat_c = 360 + np.linspace(-40, -40, 20), np.linspace(56.5, 62, 20)

        nearest_a_start = ds_slp.sel(lon=lon_a[0], lat=lat_a[0], method='nearest')
        nearest_a_end = ds_slp.sel(lon=lon_a[-1], lat=lat_a[-1], method='nearest')
        nearest_b_start = ds_slp.sel(lon=lon_b[0], lat=lat_b[0], method='nearest')
        nearest_b_end = ds_slp.sel(lon=lon_b[-1], lat=lat_b[-1], method='nearest')
        nearest_c_start = ds_slp.sel(lon=lon_c[0], lat=lat_c[0], method='nearest')
        nearest_c_end = ds_slp.sel(lon=lon_c[-1], lat=lat_c[-1], method='nearest')

        grad_a = (nearest_a_start - nearest_a_end) / \
                 g.inv(nearest_a_start.lon.values, nearest_a_start.lat.values, nearest_a_end.lon.values,
                       nearest_a_end.lat.values)[2] / 1000
        grad_b = (nearest_b_start - nearest_b_end) / \
                 g.inv(nearest_b_start.lon.values, nearest_b_start.lat.values, nearest_b_end.lon.values,
                       nearest_b_end.lat.values)[2] / 1000
        grad_c = (nearest_c_start - nearest_c_end) / \
                 g.inv(nearest_c_start.lon.values, nearest_c_start.lat.values, nearest_c_end.lon.values,
                       nearest_c_end.lat.values)[2] / 1000
        return grad_a, grad_b, grad_c

    def eof(self) -> np.ndarray:
        """
        Apply PCA on important features for Tip-Jet detection

        :return: Result of the PCA
        """
        nb_days, features = len(self.wind_mag_area.time.values), 5
        tip_jet_detection_features = np.zeros((nb_days, features))
        tip_jet_detection_features[:, 0] = self.u_at_max    # Zonal wind speed
        tip_jet_detection_features[:, 1] = self.t_at_max    # SST at max wind speed
        tip_jet_detection_features[:, 2] = self.grad_a      # Gradient pressure 1
        tip_jet_detection_features[:, 3] = self.grad_b      # Gradient pressure 2
        tip_jet_detection_features[:, 4] = self.grad_c      # Gradient pressure 3

        # Scaling the data
        scaler = StandardScaler()
        tip_jet_detection_features = scaler.fit_transform(tip_jet_detection_features)

        # we apply Principal Component Analysis
        pca = PCA(n_components=5)
        pca.fit(tip_jet_detection_features)
        pca_result = pca.transform(tip_jet_detection_features)
        eof = pca_result[:, 0]
        return eof

    def annual_events(self) -> np.ndarray:
        """
        Count the number of Tip-Jet events by year

        :return: Number of Tip-Jet events for each year in a numpy array
        """
        time_values = netCDF4.num2date(self.wind_mag_area['time'], units=self.wind_mag_area['time'].units, calendar=self.wind_mag_area['time'].calendar)

        # Create a DataFrame with time and indices
        df = pd.DataFrame({'time': time_values, 'indices': range(len(time_values))})

        tip_jet_by_year = np.zeros(2101-2015)
        for index, year in enumerate(range(2015, 2101)):
            start_date = pd.to_datetime(f'{year}-01-01')
            end_date = pd.to_datetime(f'{year+1}-01-01')

            # Use DataFrame to filter data by year and get corresponding indices
            year_indices = df.loc[(df['time'] >= start_date) & (df['time'] < end_date), 'indices'].values
            year_eof = self.eof[year_indices]
            tip_jet_by_year[index] = len(consecutive_groups(np.where(year_eof > self.threshold)[0]))
        return tip_jet_by_year

    def seasonal_events(self) -> np.ndarray:
      """
      Count the number of Tip-Jet events by season

      :return: An array of array representing the number of Tip-Jet events for each season for each year
      """
      time_values = netCDF4.num2date(self.wind_mag_area['time'], units=self.wind_mag_area['time'].units, calendar=self.wind_mag_area['time'].calendar)

      # Create a DataFrame with time and indices
      df = pd.DataFrame({'time': time_values, 'indices': range(len(time_values))})

      tip_jet_by_year = np.zeros((2101-2015,4))
      for index, year in enumerate(range(2015, 2101)):
          winter_start_date_1 = pd.to_datetime(f'{year}-01-01')
          winter_end_date_1 = pd.to_datetime(f'{year}-03-20')

          spring_start_date = pd.to_datetime(f'{year}-03-20')
          spring_end_date = pd.to_datetime(f'{year}-06-20')

          summer_start_date = pd.to_datetime(f'{year}-06-20')
          summer_end_date = pd.to_datetime(f'{year}-09-22')

          fall_start_date = pd.to_datetime(f'{year}-09-22')
          fall_end_date = pd.to_datetime(f'{year}-12-21')

          winter_start_date_2 = pd.to_datetime(f'{year}-12-21')
          winter_end_date_2 = pd.to_datetime(f'{year+1}-01-01')

          year_indices_winter = df.loc[((df['time'] >= winter_start_date_1) & (df['time'] < winter_end_date_1)) | ((df['time'] >= winter_start_date_2) & (df['time'] < winter_end_date_2)), 'indices'].values
          year_indices_spring = df.loc[(df['time'] >= spring_start_date) & (df['time'] < spring_end_date), 'indices'].values
          year_indices_summer = df.loc[(df['time'] >= summer_start_date) & (df['time'] < summer_end_date), 'indices'].values
          year_indices_fall = df.loc[(df['time'] >= fall_start_date) & (df['time'] < fall_end_date), 'indices'].values

          winter_eof = self.eof[year_indices_winter]
          spring_eof = self.eof[year_indices_spring]
          summer_eof = self.eof[year_indices_summer]
          fall_eof = self.eof[year_indices_fall]

          tip_jet_by_year[index] = np.array(
              [
                  len(consecutive_groups(np.where(winter_eof > self.threshold)[0])),
                  len(consecutive_groups(np.where(spring_eof > self.threshold)[0])),
                  len(consecutive_groups(np.where(summer_eof > self.threshold)[0])),
                  len(consecutive_groups(np.where(fall_eof > self.threshold)[0]))
                  ]
              )

      return tip_jet_by_year


class Ensemble:
    """
    Ensemble of members. Useful to do statistics on future behaviour of Tip-Jet events
    """

    def __init__(self, list_of_member, experiment):
        """
        Initialises an ensemble of members by setting its attributes

        :param list_of_member: List of member names to use for statistics
        :param experiment: Future scenarios or past data
        """
        self.list_of_member = list_of_member
        self.experiment = experiment

        # Dictionary of members
        self.members = {}
        for member_name in self.list_of_member:
            self.members[member_name] = Member(activity_id='ScenarioMIP', table_id='day', experiment_id=self.experiment,
                                          source_id='IPSL-CM6A-LR', member_id=member_name)

    def annual_events_ensemble(self) -> dict:
        """
        Count the number of Tip-Jet events for each member by year

        :return: a dictionary with the name of the member_id and the number of Tip-Jet events for each year
        """
        annual_nb_tip_jet = {}
        for member_name, member in self.members.items():
          annual_nb_tip_jet[member_name] = member.annual_events()
        return annual_nb_tip_jet

    def mean_annual_events_ensemble(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the average and standard deviation of the annual number of Tip-Jet for all members from 2015 to 2100

        :return: arrays containing the annual average and standard deviation for all the years
        """
        array_member_nb_tip_jet = np.array(list(self.annual_events_ensemble().values()))
        average_annual_nb_tip_jet = np.mean(array_member_nb_tip_jet, axis=0)
        std_annual_nb_tip_jet = np.std(array_member_nb_tip_jet, axis=0)
        return average_annual_nb_tip_jet, std_annual_nb_tip_jet

    def seasonal_events_ensemble(self) -> dict:
        """
        Count the number of Tip-Jet events for each member and for each season of each year

        :return: a dictionary with the name of the member_id and the number of Tip-Jet events for each season of each year
        """
        seasonal_nb_tip_jet = {}
        for member_name, member in self.members.items():
          seasonal_nb_tip_jet[member_name] = member.seasonal_events()
        return seasonal_nb_tip_jet

    def mean_seasonal_events_ensemble(self) -> np.ndarray:
        """
        average the seasonal number of Tip-Jet for all members from 2015 to 2100

        :return: an array containing the seasonal average number of Tip-jet for all the members
        """
        array_member_nb_tip_jet = np.array(list(self.seasonal_events_ensemble().values()))
        average_seasonal_nb_tip_jet = np.mean(array_member_nb_tip_jet, axis=0)
        std_seasonal_nb_tip_jet = np.std(array_member_nb_tip_jet, axis=0)
        return average_seasonal_nb_tip_jet, std_seasonal_nb_tip_jet