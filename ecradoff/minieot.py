import os
import numpy as np

# IFS data
IFSDATADIR = "../data"

# RDAY
DAYSECS = 86400
# days in 1 year
YEADAYS = 365.25

# mean earth-sun distance
REA     = 1.

# polar axis tilting
REPSM   = 0.409093

# solar irradiance dset
IFSIRRVERS =  "49r1"
SOLIRRDSET = os.path.join(IFSDATADIR, f"total_solar_irradiance_CMIP6_{IFSIRRVERS}.nc")


# Class Irradiance
class Irradiance:
    def __init__(self, date : np.datetime64, delay_s : float = 0., ifs_like_2pi : bool = True,
                 ignore_eot : bool = False, skip_irr : bool = False):
        if not isinstance(date, np.datetime64):
            raise ValueError("date must be of type np.datetime64!")

        def _year_fraction(self) -> float:
            """ Computes the fraction of the yearly Earth's orbit
            around the sun from the start of the year to date
            RTETA
            """
            this_yr        = self.date.astype("datetime64[Y]")
            return (self.date - this_yr)/(np.timedelta64(24, "h")*YEADAYS)

        def _day_fraction(self) -> float:
            """ Computes the fraction of the day at the current date
            """
            today = self.date.astype("datetime64[D]")
            return (self.date - today)/np.timedelta64(24, "h")

        def _get_solar_irr(self) -> float:
            import xarray as xr
            sol_irr_dset = xr.open_dataset(SOLIRRDSET, decode_times=False)["tsi"]

            this_yr      = self.date.astype("datetime64[Y]").astype(int)+1970
            now_yr       = this_yr + self.year_fraction

            sun_irr      = float(sol_irr_dset.interp(time=now_yr, method="linear"))

            return sun_irr*(1/self.earth_sun_dist_frac**2)


        def _orbit_theta(self) -> float:
            """ Computes the angle of the yearly
            Earth's orbit around the sun to date (no precession of equinoxes)
            REL reference
            """
            two_pi = 6.283076 if ifs_like_2pi else 2*np.pi

            return 1.7535 + two_pi*self.year_fraction

        def _orbit_theta_rem(self) -> float:
            """ Computes the  angle of the yearly
            Earth's orbit around the sun to date (no precession of equinoxes)
            REM reference
            """
            two_pi = 6.283020 if ifs_like_2pi else 2*np.pi

            return 6.240075 + two_pi*self.year_fraction

        def _earth_sun_dist_frac(self) -> float:
            """ Computes the distance between the Earth and the sun
            for the current date
            date : np.datetime64 or np.ndarray of np.datetime64
            RRS reference as a fraction of REA (earth-sun distance mean)
            """
            return 1.0001 - 0.0163*np.sin(self.orbit_theta) + 0.0037*np.cos(self.orbit_theta)

        # RDS equivalent to eq. 3.7 below
        def _sun_declination(self) -> float:
            """ Computes the sun declination for the current date
            date : np.datetime64 or np.ndarray of np.datetime64
            """
            two_pi = 6.283320 if ifs_like_2pi else 2*np.pi
            relative_rllls = 4.8952 + two_pi*self.year_fraction - 0.0075*np.sin(self.orbit_theta) +\
                -0.0326*np.cos(self.orbit_theta) - 0.0003*np.sin(2*self.orbit_theta)              +\
                +0.0002*np.cos(2*self.orbit_theta)
            return np.arcsin(np.sin(REPSM)*np.sin(relative_rllls))

        def _equation_of_time_s(self) -> float:
            """ Equation of time for the current date (in seconds)
            date : np.datetime64 or np.ndarray of np.datetime64
            """
            two_pi = 6.283076 if ifs_like_2pi else 2*np.pi
            relative_rlls = 4.8951 + two_pi*self.year_fraction

            sin_rem       = np.sin(self.orbit_theta_rem)
            return 591.8*np.sin(2*relative_rlls) - 459.4*sin_rem +\
                +39.5*sin_rem*np.cos(2*relative_rlls)            +\
                -12.7*np.sin(4*relative_rlls) - 4.8*np.sin(2*self.orbit_theta_rem)

        def _solar_time(self) -> float:
            """ Computes the solar time at the current date in radians
            date : np.datetime64 or np.ndarray of np.datetime64
            """
            return 2*np.pi*(self.eq_of_time_s/DAYSECS + self.day_fraction)

        # Set date
        self.date            = date - np.timedelta64(1,"s")*delay_s

        # Fill astronomical data
        self.year_fraction   = _year_fraction(self)
        self.day_fraction    = _day_fraction(self)

        # Require self.year_fraction
        self.orbit_theta     = _orbit_theta(self)
        self.orbit_theta_rem = _orbit_theta_rem(self)

        # Require self.orbit_theta
        self.earth_sun_dist_frac = _earth_sun_dist_frac(self)
        self.earth_sun_dist_m    = REA*self.earth_sun_dist_frac
        self.sun_declination_rad = _sun_declination(self)
        self.sun_declination_deg = np.rad2deg(self.sun_declination_rad)

        # Require self.orbit_theta_rem (option to ignore equation of time)
        self.eq_of_time_s    = 0. if ignore_eot else _equation_of_time_s(self)

        # Requires self.eq_of_time_s
        self.solar_time      = _solar_time(self)

        # Require year_fraction and earth_sun_dist_frac
        if not skip_irr:
            self.solar_irr       = _get_solar_irr(self)


    def mu0_cos_sza_rad(self, phi : float, lam : float, zamu0 : bool = False) -> float:
        """ Computes the cosine of solar zenith angle at the current date
        date  : np.datetime64 or np.ndarray of np.datetime64
        phi   : float or np.ndarray latitude in radians
        lam   : float or np.ndarray longitude in radians
        zamu0 : bool set to true to simulate IFS input to ecrad
        """


        decl       = self.sun_declination_rad
        h_angle    = self.solar_time + lam + np.pi
        mu0        = np.sin(decl)*np.sin(phi) + \
                    + np.cos(decl)*np.cos(phi)*np.cos(h_angle)
        if zamu0:
            rrae = 0.1277*1.e-2
            zcrae = rrae*(rrae+2)
            mu0[...] = np.where(mu0>1.e-10, rrae/(np.sqrt(mu0**2+zcrae)-mu0), rrae/np.sqrt(zcrae))

        return np.clip(mu0, 0., 1.)

    def mu0_cos_sza_deg(self, phi : float, lam : float, zamu0 : bool = False) -> float:
        """ Computes the cosine of solar zenith angle at the current date
        date : np.datetime64 or np.ndarray of np.datetime64
        phi  : float or np.ndarray latitude in degrees
        lam  : float or np.ndarray longitude in degrees
        zamu0 : bool set to true to simulate IFS input to ecrad
        """

        return self.mu0_cos_sza_rad(phi=np.deg2rad(phi), lam=np.deg2rad(lam), zamu0=zamu0)
