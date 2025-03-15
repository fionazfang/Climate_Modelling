# import math
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # Load the insolation data (adjust the path if needed)
# insolation_data = pd.read_excel(r'D:\OneDrive - University of Cambridge\学术蓄水\Research!\日照模型\merged_insolation_data.xlsx')
#
# class ClimateModel:
#     def __init__(self, A=204, B=2.17, C=3.87, Aice=0.62, Acloud=0.3, Tcrit=-10, SCfrac=1, SC=1370, iterations=30):
#         self.A = A
#         self.B = B
#         self.C = C
#         self.Aice = Aice
#         self.Acloud = Acloud
#         self.Tcrit = Tcrit
#         self.SCfrac = SCfrac
#         self.SC = SC
#         self.iterations = iterations
#
#         self.Zones = ["80-90", "70-80", "60-70", "50-60", "40-50", "30-40", "0-30", "0-20", "0-10"]
#         self.ZoneLat = [85, 75, 65, 55, 45, 35, 25, 15, 6]
#         self.ZoneSun = [0.5, 0.531, 0.624, 0.77, 0.892, 1.021, 1.12, 1.189, 1.219]
#         self.Init_T = [-16.9, -12.3, -5.1, 2.2, 8.8, 16.2, 22.9, 26.1, 26.4]
#
#         self.sumcos, self.coz = self.calculate_cosine_weighting()
#         self.R_in = [self.SC / 4 * self.SCfrac * self.ZoneSun[lat] for lat in range(9)]
#
#         self.Mean_T = []
#         self.Albedo = []
#         self.Temp = []
#         self.R_out = []
#
#     def calculate_cosine_weighting(self):
#         sumcos = 0
#         coz = []
#         for lat in range(9):
#             coz_val = math.cos(self.ZoneLat[lat] * math.pi / 180)
#             coz.append(coz_val)
#             sumcos += coz_val
#         return sumcos, coz
#
#     def run_simulation(self):
#         temp_step = self.Init_T[:]
#         albedo_step = [(self.Aice if T < self.Tcrit else self.Acloud) for T in self.Init_T]
#         self.Temp.append(temp_step[:])
#         self.Albedo.append(albedo_step[:])
#         r_out_step = []
#
#         for step in range(self.iterations + 1):
#             Tcos = [temp * self.coz[lat] for lat, temp in enumerate(temp_step)]
#             mean_temp = sum(Tcos) / self.sumcos
#             self.Mean_T.append(mean_temp)
#
#             for lat in range(9):
#                 temp_step[lat] = (self.R_in[lat] * (1 - albedo_step[lat]) + self.C * mean_temp - self.A) / (self.B + self.C)
#                 albedo_step[lat] = self.Aice if temp_step[lat] < self.Tcrit else self.Acloud
#                 r_out_step.append(self.A + self.B * temp_step[lat])
#
#             self.Temp.append(temp_step[:])
#             self.Albedo.append(albedo_step[:])
#             self.R_out.append(r_out_step[:])
#
#     def plot_global_mean_temperature(self):
#         plt.figure()
#         plt.plot(self.Mean_T, label="Global Mean Temperature")
#         plt.xlabel("Time Step")
#         plt.ylabel("Temperature (°C)")
#         plt.title("Global Mean Temperature over Time")
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#
# # Subclass to handle real insolation data
# class RealInsolationModel(ClimateModel):
#     def __init__(self, insolation_data, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.insolation_data = insolation_data
#         self.iterations = len(insolation_data)
#
#     def update_insolation(self, time_index):
#         insolation_row = self.insolation_data.iloc[time_index]
#         self.R_in = [
#             insolation_row['Lat_85'],  # 85°
#             insolation_row['Lat_75'],  # 75°
#             insolation_row['Lat_65'],  # 65°
#             insolation_row['Lat_55'],  # 55°
#             insolation_row['Lat_45'],  # 45°
#             insolation_row['Lat_35'],  # 35°
#             insolation_row['Lat_25'],  # 25°
#             insolation_row['Lat_15'],  # 15°
#             insolation_row['Lat_5']    # 5°
#         ]
#
#     def run_simulation_with_real_insolation(self):
#         for step in range(self.iterations):
#             self.update_insolation(step)
#             super().run_simulation()
#
# # Running the model
# real_insolation_model = RealInsolationModel(insolation_data)
# real_insolation_model.run_simulation_with_real_insolation()
# real_insolation_model.plot_global_mean_temperature()



from kgcpy import Koeppen

# Example coordinates (latitude and longitude) to find climate zone
latitude = 45.0   # Change this to your specific latitude
longitude = -93.0 # Change this to your specific longitude

# Initialize the Koeppen class
climate_classifier = Koeppen()

# Get the climate zone for the specific point
climate_zone = climate_classifier.koeppen_point(latitude, longitude)

print(f"The climate zone at latitude {latitude} and longitude {longitude} is: {climate_zone}")
