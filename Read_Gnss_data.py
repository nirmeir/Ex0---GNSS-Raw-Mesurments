
import sys, os, csv
parent_directory = os.path.split(os.getcwd())[0]
ephemeris_data_directory = os.path.join(parent_directory, 'data')
sys.path.insert(0, parent_directory)
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import navpy
from gnssutils import EphemerisManager
from Positioning_Algorithm import Positioning_Algorithm
from EcefTolla import ecef_to_lla
import simplekml



WEEKSEC = 604800
LIGHTSPEED = 2.99792458e8


def calculate_satellite_position(ephemeris,timestamp,one_epoch):
    kml = simplekml.Kml()

    transmit_times = one_epoch['tTxSeconds']
    mu = 3.986005e14
    OmegaDot_e = 7.2921151467e-5
    F = -4.442807633e-10

    sv_position = pd.DataFrame(index=ephemeris.index)

    sv_position['GPS time'] = timestamp
    sv_position['SatPRN (ID)'] = ephemeris['sv']

    sv_position.set_index('SatPRN (ID)', inplace=True)


    ephemeris.set_index('sv', inplace=True)

    sv_position['t_k'] = transmit_times - ephemeris['t_oe']
    # print(sv_position['t_k'])

    A = ephemeris['sqrtA'] ** 2
    n_0 = np.sqrt(mu / A ** 3)
    n = n_0 + ephemeris['deltaN']
    M_k = ephemeris['M_0'] + n * sv_position['t_k']
    E_k = M_k
    err = pd.Series(data=[1]*len(sv_position.index))
    i = 0

    while err.abs().min() > 1e-8 and i < 10:
        new_vals = M_k + ephemeris['e']*np.sin(E_k)
        err = new_vals - E_k
        E_k = new_vals
        i += 1

    sinE_k = np.sin(E_k)
    cosE_k = np.cos(E_k)
    delT_r = F * ephemeris['e'] * ephemeris['sqrtA'] * sinE_k
    delT_oc = transmit_times - ephemeris['t_oc']

    sv_position['delT_sv'] = ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc + ephemeris['SVclockDriftRate'] * delT_oc ** 2

    v_k = np.arctan2(np.sqrt(1 - ephemeris['e'] ** 2) * sinE_k, cosE_k - ephemeris['e'])
    Phi_k = v_k + ephemeris['omega']
    sin2Phi_k = np.sin(2 * Phi_k)
    cos2Phi_k = np.cos(2 * Phi_k)

    du_k = ephemeris['C_us'] * sin2Phi_k + ephemeris['C_uc'] * cos2Phi_k
    dr_k = ephemeris['C_rs'] * sin2Phi_k + ephemeris['C_rc'] * cos2Phi_k
    di_k = ephemeris['C_is'] * sin2Phi_k + ephemeris['C_ic'] * cos2Phi_k

    u_k = Phi_k + du_k
    r_k = A * (1 - ephemeris['e'] * cosE_k) + dr_k
    i_k = ephemeris['i_0'] + di_k + ephemeris['IDOT'] * sv_position['t_k']

    x_k_prime = r_k * np.cos(u_k)
    y_k_prime = r_k * np.sin(u_k)

    Omega_k = ephemeris['Omega_0'] + (ephemeris['OmegaDot'] - OmegaDot_e) * sv_position['t_k'] - OmegaDot_e * ephemeris['t_oe']

    sv_position['Sat.X'] = x_k_prime * np.cos(Omega_k) - y_k_prime * np.cos(i_k) * np.sin(Omega_k)
    sv_position['Sat.Y'] = x_k_prime * np.sin(Omega_k) + y_k_prime * np.cos(i_k) * np.cos(Omega_k)
    sv_position['Sat.Z'] = y_k_prime * np.sin(i_k)

    b0 = 0
    x0 = np.array([0, 0, 0])
    xs = sv_position[['Sat.X', 'Sat.Y', 'Sat.Z']].to_numpy()
    print(len(xs))

    # Apply satellite clock bias to correct the measured pseudorange values
    pr = one_epoch['PrM'] + LIGHTSPEED * sv_position['delT_sv']
    pr = pr.to_numpy()
    sv_position['Pseudo-Range'] = pr
    sv_position['CN0'] = one_epoch['Cn0DbHz']


    x, b, dp = Positioning_Algorithm(xs, pr, x0, b0)
    sv_position['POS.X'] = x[0]
    sv_position['POS.Y'] = x[1] 
    sv_position['POS.Z'] = x[2] 
    print("x: ",x)

    sv_position['Lat'],sv_position['Lon'],sv_position['Alt'] = ecef_to_lla(x[0],x[1],x[2])
    print(navpy.ecef2lla(x))
    
    list_of_coordinates = []
  
    for row in range (len(sv_position['Lat'])):
        print(sv_position['Lat'].iloc[row])
        print(sv_position['Lon'].iloc[row])
        print(sv_position['Alt'].iloc[row])
        list_of_coordinates.append((sv_position['Lon'].iloc[row],sv_position['Lat'].iloc[row],sv_position['Alt'].iloc[row]))

    for coord in list_of_coordinates:
        point = kml.newpoint(coords=[coord])
        point.name = f"Point at {coord[0]}, {coord[1]}"

    print(list_of_coordinates)
    kml.save("KML.kml")
    

    return sv_position


def read_data(input_filepath):

    # Function to parse GNSS log data and filter for GPS satellites
    with open(input_filepath) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0][0] == '#':
                if 'Fix' in row[0]:
                    android_fixes = [row[1:]]
                elif 'Raw' in row[0]:
                    measurements = [row[1:]]
            else:
                if row[0] == 'Fix':
                    android_fixes.append(row[1:])
                elif row[0] == 'Raw':
                    measurements.append(row[1:])

    android_fixes = pd.DataFrame(android_fixes[1:], columns = android_fixes[0])
    measurements = pd.DataFrame(measurements[1:], columns = measurements[0])

    # Format satellite IDs
    measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']
    measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
    measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'
    measurements['SvName'] = measurements['Constellation'] + measurements['Svid']

    # Remove all non-GPS measurements
    measurements = measurements.loc[measurements['Constellation'] == 'G']

    # Convert columns to numeric representation
    measurements['Cn0DbHz'] = pd.to_numeric(measurements['Cn0DbHz'])
    measurements['TimeNanos'] = pd.to_numeric(measurements['TimeNanos'])
    measurements['FullBiasNanos'] = pd.to_numeric(measurements['FullBiasNanos'])
    measurements['ReceivedSvTimeNanos']  = pd.to_numeric(measurements['ReceivedSvTimeNanos'])
    measurements['PseudorangeRateMetersPerSecond'] = pd.to_numeric(measurements['PseudorangeRateMetersPerSecond'])
    measurements['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(measurements['ReceivedSvTimeUncertaintyNanos'])

    # A few measurement values are not provided by all phones
    # We'll check for them and initialize them with zeros if missing
    if 'BiasNanos' in measurements.columns:
        measurements['BiasNanos'] = pd.to_numeric(measurements['BiasNanos'])
    else:
        measurements['BiasNanos'] = 0
    if 'TimeOffsetNanos' in measurements.columns:
        measurements['TimeOffsetNanos'] = pd.to_numeric(measurements['TimeOffsetNanos'])
    else:
        measurements['TimeOffsetNanos'] = 0

    # print(measurements.columns)


    measurements['GpsTimeNanos'] = measurements['TimeNanos'] - (measurements['FullBiasNanos'] - measurements['BiasNanos'])
    gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
    measurements['UnixTime'] = pd.to_datetime(measurements['GpsTimeNanos'], utc = True, origin=gpsepoch)
    measurements['UnixTime'] = measurements['UnixTime']

    # Split data into measurement epochs
    measurements['Epoch'] = 0
    measurements.loc[measurements['UnixTime'] - measurements['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    measurements['Epoch'] = measurements['Epoch'].cumsum()


    # This should account for rollovers since it uses a week number specific to each measurement

    measurements['tRxGnssNanos'] = measurements['TimeNanos'] + measurements['TimeOffsetNanos'] - (measurements['FullBiasNanos'].iloc[0] + measurements['BiasNanos'].iloc[0])
    measurements['GpsWeekNumber'] = np.floor(1e-9 * measurements['tRxGnssNanos'] / WEEKSEC)
    measurements['tRxSeconds'] = 1e-9*measurements['tRxGnssNanos'] - WEEKSEC * measurements['GpsWeekNumber']
    measurements['tTxSeconds'] = 1e-9*(measurements['ReceivedSvTimeNanos'] + measurements['TimeOffsetNanos'])
    # Calculate pseudorange in seconds
    measurements['prSeconds'] = measurements['tRxSeconds'] - measurements['tTxSeconds']

    measurements['PrM'] = LIGHTSPEED * measurements['prSeconds']
    measurements['PrSigmaM'] = LIGHTSPEED * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos']


    manager = EphemerisManager(ephemeris_data_directory)

    epoch = 0
    num_sats = 0

    while num_sats < 5 :
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)].drop_duplicates(subset='SvName')
        timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
        one_epoch.set_index('SvName', inplace=True)
        num_sats = len(one_epoch.index)
        epoch += 1

    sats = one_epoch.index.unique().tolist()
    ephemeris = manager.get_ephemeris(timestamp, sats)
    sv_position = calculate_satellite_position(ephemeris,timestamp,one_epoch)

      
    output_path = "satellite_positions.csv"

    sv_position = sv_position.drop(columns=['t_k'])
    sv_position = sv_position.drop(columns=['delT_sv'])

    sv_position.to_csv(output_path, index=True)






