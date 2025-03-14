import xarray as xr
from datetime import datetime
import numpy as np
import json
from scipy.stats import circmean, circstd
import numpy as np
from IHSetUtils import abs_pos



class output_standard_netCDF(object):
    """
    output_standard_netCDF
    
    Save the simulations in the standard format for IH-SET.
    
    This class reads input datasets.
    """
    def __init__(self, path, model):
        # Dimensions
        self.path = path
        self.model = model
        self.mode = model.mode
        self.type = model.type
        self.ds = xr.open_dataset(path)
        self.ds.load()
        self.ds.close()
        self.inp_filename = path.split('/')[-1]

        self.filename = 'out_'+self.inp_filename.split('.')[0]+'.nc'
        self.path = path.replace(self.inp_filename, self.filename)

        # Check if output file already exists
        if self.filename in path:
            print(f"Output file {self.filename} already exists. Loading it to add new results.")
            self.writing_flag = False
            self.write_to_output_file()
        else:
            print(f"Creating output file {self.filename}")
            self.writing_flag = True
            self.create_output_file()

               
    def set_attrs(self):
        """ Set global attributes """
        # Global attributes

        # data_sources = f'Waves: {self.w_dataSource}, Surge: {self.dataSource_surge}, Tide: {self.dataSource_tide}, Obs: {self.obs_dataSource}'

        # Lets use the date with YYY-MM-DDThh:mm:ssZ format        
        creation_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        simulations_info = {"1": {"Model Name": self.model.name, "Mode": self.mode, "Configuration": self.model.cfg}}
        # transform it to json format
        simulations_info = json.dumps(simulations_info)

        self.check_models()

        self.attrs = {
            "title": "Output IH-SET file",
            "institution": "Environmental Hydraulics Institute of Cantabria - https://ihcantabria.com/",
            "source": "IH-SET",
            "history": f'Created on {creation_date}',
            "references": "Jaramillo et al. (2025) - doi: xxxxxxxx.xx",
            "Documentation": "https://ihcantabria.github.io/IHSetDocs/",
            "Conventions": "CF-1.6",
            # "Data Sources": data_sources,
            "summary": "This dataset is output from IH-SET Simulations. Etc…",
            "geospatial_lat_min": -90,
            "geospatial_lat_max": 90,
            "geospatial_lon_min": -180,
            "geospatial_lon_max": 180,
            "input_file": self.inp_filename,
            "EPSG": self.ds.EPSG.values,
            "simulations": simulations_info,
        }

    def create_output_file(self):
        """
        Create the output file
        """

        self.set_attrs()
        self.transform_data()
        self.set_simulation_attrs()



        # Create dataset with xarray
        ds = xr.Dataset(
            {
                "obs": (("time_obs", "ntrs"), self.ds.obs, self.ds.obs.attrs),
                "rot": ("time_obs", self.ds.rot, self.ds.rot.attrs),
                "average_obs": ("time_obs", self.ds.average_obs, self.ds.average_obs.attrs),

            },
            coords={
                "time": ("time", self.ds.time, {
                    "standard_name": "time",
                    "long_name": "Time"
                }),
                "time_1": ("time_1", self.model.time, {
                    "standard_name": "time_1",
                    "long_name": "Time for model 1"
                }),
                "ntrs": ("ntrs",self.ds.ntrs, {
                    "units": "number_of",
                    "standard_name": "number_of_trs",
                    "long_name": "Number of Transects"
                }),
                "time_obs": ("time_obs", self.ds.time_obs, {
                    "standard_name": "time_of_observations",
                    "long_name": "Time of Observation"
                }),
                "xi": ("xi", self.ds.xi, {
                    "units": "meters",
                    "standard_name": "xi_coordinate",
                    "long_name": "Origin x coordinate of transect"
                }),
                "yi": ("yi", self.ds.yi, {
                    "units": "meters",
                    "standard_name": "yi_coordinate",
                    "long_name": "Origin y coordinate of transect"
                }),
                "xf": ("xf", self.ds.xf, {
                    "units": "meters",
                    "standard_name": "xf_coordinate",
                    "long_name": "End x coordinate of transect"
                }),
                "yf": ("yf", self.ds.yf, {
                    "units": "meters",
                    "standard_name": "yf_coordinate",
                    "long_name": "End y coordinate of transect"
                }),
                "phi": ("phi", self.ds.phi, {
                    "units": "degrees",
                    "standard_name": "transect_angle",
                    "long_name": "Cartesian angle of transect"
                }),
                "x_pivotal": ("x_pivotal", self.ds.x_pivotal, {
                    "units": "meters",
                    "standard_name": "x_pivotal",
                    "long_name": "Initial x coordinate of pivotal transect"
                }),
                "y_pivotal": ("y_pivotal", self.ds.y_pivotal, {
                    "units": "meters",
                    "standard_name": "y_pivotal",
                    "long_name": "Initial y coordinate of pivotal transect"
                }),
                "phi_pivotal": ("phi_pivotal", self.ds.phi_pivotal, {
                    "units": "degrees",
                    "standard_name": "phi_pivotal",
                    "long_name": "Angle of pivotal transect"
                }),

            },
            attrs=self.attrs
        )

        if self.type == 'CS':
            ds["simulation_1"] = (("time_1"), self.model.full_run, self.simulation_attrs)
        elif self.type == 'RT':
            ds["simulation_1"] = (("time_1"), self.model.full_run, self.simulation_attrs)
        elif self.type == 'HY':
            ds["simulation_1"] = (("time_1", "ntrs"), self.model.full_run, self.simulation_attrs)
            ds["simulation_1_avg"] = (("time_1"), np.mean(self.model.full_run, axis=1), self.simulation_attrs)
            rot = calculate_rotation(self.ds.xi, self.ds.yi, self.ds.phi, self.model.full_run)
            ds["simulation_1_rot"] = (("time_1"), rot, self.simulation_attrs)
        elif self.type == 'OL':
            ds["simulation_1"] = (("time_1", "ntrs"), self.model.full_run, self.simulation)
            ds["simulation_1_avg"] = (("time_1"), np.mean(self.model.full_run, axis=1), self.simulation_attrs)
            rot = calculate_rotation(self.ds.xi, self.ds.yi, self.ds.phi, self.model.full_run)
            ds["simulation_1_rot"] = (("time_1"), rot, self.simulation_attrs)
        
        # Export to NetCDF
        ds.to_netcdf(self.path, engine="netcdf4")

        print(f"{self.filename} saved correctly.")

    def write_to_output_file(self):
        """
        Write new data to the output file
        """

        self.transform_data()
        self.set_simulation_attrs()

        # Open the output file
        ds = xr.open_dataset(self.path, mode="a")

        # first we need to check how many simulation already exist
        n_sim = len(ds.data_vars) - 3

        
        # Add the coordinate f"time_{n_sim+1}" to the dataset
        ds.coords[f"time_{n_sim+1}"] = self.model.time

        if self.type == 'CS':
            ds[f"simulation_{n_sim+1}"] = ((f"time_{n_sim+1}"), self.model.full_run, self.simulation_attrs)
        elif self.type == 'RT':
            ds[f"simulation_{n_sim+1}"] = ((f"time_{n_sim+1}"), self.model.full_run, self.simulation_attrs)
        elif self.type == 'HY':
            ds[f"simulation_{n_sim+1}"] = ((f"time_{n_sim+1}", "ntrs"), self.model.full_run, self.simulation_attrs)
            ds[f"simulation_{n_sim+1}_avg"] = ((f"time_{n_sim+1}"), np.mean(self.model.full_run, axis=1), self.simulation_attrs)
            rot = calculate_rotation(self.ds.xi, self.ds.yi, self.ds.phi, self.model.full_run)
            ds[f"simulation_{n_sim+1}_rot"] = ((f"time_{n_sim+1}"), rot, self.simulation_attrs)
        elif self.type == 'OL':
            ds[f"simulation_{n_sim+1}"] = ((f"time_{n_sim+1}", "ntrs"), self.model.full_run, self.simulation)
            ds[f"simulation_{n_sim+1}_avg"] = ((f"time_{n_sim+1}"), np.mean(self.model.full_run, axis=1), self.simulation_attrs)
            rot = calculate_rotation(self.ds.xi, self.ds.yi, self.ds.phi, self.model.full_run)
            ds[f"simulation_{n_sim+1}_rot"] = ((f"time_{n_sim+1}"), rot, self.simulation_attrs)

        # Save the dataset
        ds.to_netcdf(self.path, engine="netcdf4")

        ds.close()

        print(f"New data saved to {self.filename}.")

    def set_simulation_attrs(self):
        """ Set model attributes """

        if self.type == 'CS':
            self.simulation_attrs = {
                "units": "Meters",
                "standard_name": "shoreline_position",
                "model_type": "Cross shore",
                "long_name": f"Shoreline position calulated by the model{self.model.name}",
                "max_value": np.nanmax(self.model.full_run),
                "min_value": np.nanmin(self.model.full_run),
                "mean_value": np.nanmean(self.model.full_run),
                "standard_deviation": np.nanstd(self.model.full_run),
                "transect": self.model.cfg["trs"]
            }
        elif self.type == 'RT':
            self.simulation_attrs = {
                "units": "Meters",
                "standard_name": "shoreline_orientation",
                "model_type": "Rotation",
                "long_name": f"Shoreline position calulated by the model{self.model.name}",
                "max_value": np.nanmax(self.model.full_run),
                "min_value": np.nanmin(self.model.full_run),
                "mean_value": circmean(self.model.full_run, high=360, low=0),
                "standard_deviation": circstd(self.model.full_run, high=360, low=0),
                "transect": self.model.cfg["trs"]
            }
        elif self.type == 'HY':
            self.simulation_attrs = {
                "units": "Meters",
                "standard_name": "shoreline_position",
                "model_type": "Hybrid",
                "long_name": f"Shoreline position calulated by the model{self.model.name}",
            }
        elif self.type == 'OL':
            self.simulation_attrs = {
                "units": "Meters",
                "standard_name": "shoreline_position",
                "model_type": "One Line",
                "long_name": f"Shoreline position calulated by the model{self.model.name}",
            }


        for key, value in key(self.model.par_names, self.model.par_values):
            self.simulation_attrs[key] = value


    def transform_data(self):
        """ Transform shoreline positions to projected coordinates """

        self.x, self.y = abs_pos(self.ds.xi, self.ds.yi, self.ds.phi, self.full_run)



def calculate_rotation(X0, Y0, phi, dist):
    """
    Calculate the shoreline rotation.
    """

    phi_rad = np.deg2rad(phi)

    mean_shoreline = np.nanmean(dist, axis=0)

    detrended_dist = np.zeros(dist.shape)

    for i in range(dist.shape[1]):
        detrended_dist[:, i] = dist[:, i] - mean_shoreline[i]

    # We will calculate the rotation only for the times where we at least 80% of the data

    nans_rot = np.sum(np.isnan(detrended_dist), axis=1) > 0.2 * dist.shape[1]
    
    alpha = np.zeros(dist.shape[0]) * np.nan
    
    for i in range(dist.shape[0]):
        if not nans_rot[i]:
            XN, YN = X0 + detrended_dist[i, :] * np.cos(phi_rad), Y0 + detrended_dist[i, :] * np.sin(phi_rad)
            ii_nan = np.isnan(XN) | np.isnan(YN)
            fitter = np.polyfit(XN[~ii_nan], YN[~ii_nan], 1)
            alpha[i] = 90 - np.rad2deg(np.arctan(fitter[0]))
            if alpha[i] < 0:
                alpha[i] += 360

    mask_nans = np.isnan(alpha)

    # mean_alpha_ori = circmean(alpha[~mask_nans], high=360, low=0)
    # if mean_alpha_ori<0:
    #     mean_alpha_ori += 360

    mean_phi = 90 - circmean(phi, high=360, low=0)
    if mean_phi<0:
        mean_phi += 360
    mean_alpha = circmean(alpha[~mask_nans], high=360, low=0) + 90
    if mean_alpha<0:
        mean_alpha += 360   
    mean_alpha_2 = circmean(alpha[~mask_nans], high=360, low=0) - 90
    if mean_alpha_2<0:
        mean_alpha_2 += 360

    # print(f"Mean alpha: {mean_alpha}, Mean phi: {mean_phi}, Mean alpha 2: {mean_alpha_2}, Mean Alpha_ori: {mean_alpha_ori}")

    if np.abs(mean_alpha - mean_phi) <= np.abs(mean_alpha_2 - mean_phi) :
        alpha  += 90
    else:
        alpha -= 90
    
    # Now we change the <0 to 0-360

    alpha[alpha < 0] += 360

    return alpha, mask_nans
