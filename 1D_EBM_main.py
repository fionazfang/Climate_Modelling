# √ 1.让每次迭代都记录过程值 (a, R_in, R_out, MeanT, Temp)
# √ 2.visualise时间序列
# √ 3.封装进class
# √ 4-1.1ma/1.5ma - 更新insolation时间序列 - 看T和ice cover与实际比较
# 4-2.1ma/1.5ma - 更换albedo，去掉随温度迭代部分
# 4-3.1ma/1.5ma - 更新GHG时间序列（查找资料）
# 5.Holocene - 更新insolation，GHG，ice cover，太阳活动，火山活动




import math
import matplotlib.pyplot as plt
import pandas as pd

insolation_data = pd.read_excel(r'D:\OneDrive - University of Cambridge\学术蓄水\Research!\日照模型\merged_insolation_data.xlsx')

class EnergyBalanceModel:
    def __init__(self, A=204, B=2.17, C=3.87, Aice=0.62, Acloud=0.3, Tcrit=-10, SCfrac=1, SC=1370, iterations=30):
        self.A = A
        self.B = B
        self.C = C
        self.Aice = Aice
        self.Acloud = Acloud
        self.Tcrit = Tcrit
        self.SCfrac = SCfrac
        self.SC = SC
        self.iterations = iterations

        self.Zones = ["80-90", "70-80", "60-70", "50-60", "40-50", "30-40", "0-30", "0-20", "0-10"]
        self.ZoneLat = [85, 75, 65, 55, 45, 35, 25, 15, 6]
        self.ZoneSun = [0.5, 0.531, 0.624, 0.77, 0.892, 1.021, 1.12, 1.189, 1.219]
        self.Init_T = [-16.9, -12.3, -5.1, 2.2, 8.8, 16.2, 22.9, 26.1, 26.4]

        self.sumcos, self.coz = self.calculate_cosine_weighting()
        self.R_in = [self.SC / 4 * self.SCfrac * self.ZoneSun[lat] for lat in range(9)]

        self.Mean_T = []
        self.Albedo = []
        self.Temp = []

    def calculate_cosine_weighting(self):
        sumcos = 0
        coz = []
        for lat in range(9):
            coz_val = math.cos(self.ZoneLat[lat] * math.pi / 180)
            coz.append(coz_val)
            sumcos += coz_val
        return sumcos, coz

    def run_simulation(self):
        temp_step = self.Init_T[:]
        albedo_step = [(self.Aice if T < self.Tcrit else self.Acloud) for T in self.Init_T]
        self.Temp.append(temp_step[:])
        self.Albedo.append(albedo_step[:])

        for step in range(self.iterations + 1):
            Tcos = [temp * self.coz[lat] for lat, temp in enumerate(temp_step)]
            mean_temp = sum(Tcos) / self.sumcos
            self.Mean_T.append(mean_temp)

            for lat in range(9):
                temp_step[lat] = (self.R_in[lat] * (1 - albedo_step[lat]) + self.C * mean_temp - self.A) / (self.B + self.C)
                albedo_step[lat] = self.Aice if temp_step[lat] < self.Tcrit else self.Acloud

            self.Temp.append(temp_step[:])
            self.Albedo.append(albedo_step[:])

    def plot_global_mean_temperature(self):
        """Plot the global mean temperature over time."""
        plt.figure()
        plt.plot(self.Mean_T, label="Global Mean Temperature", linestyle='-')
        plt.xlabel("Time Step")
        plt.ylabel("Temperature (°C)")
        plt.title("Global Mean Temperature over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_temperature_evolution(self, latitude_input):
        """Plot temperature evolution for a specific latitude requested by the user."""
        try:
            zone_index = self.ZoneLat.index(latitude_input)
        except ValueError:
            print(f"Latitude {latitude_input}° not available. Please choose from {self.ZoneLat}")
            return

        temperature_series = [self.Temp[step][zone_index] for step in range(len(self.Temp))]

        plt.figure()
        plt.plot(range(len(temperature_series)), temperature_series, label=f"Temperature at {latitude_input}°",
                 linestyle='-', color='orange')
        plt.xlabel("Time Step")
        plt.ylabel("Temperature (°C)")
        plt.title(f"Temperature Evolution at Latitude {latitude_input}°")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_albedo_evolution(self, latitude_input):
        """Plot albedo evolution for a specific latitude requested by the user."""
        try:
            zone_index = self.ZoneLat.index(latitude_input)
        except ValueError:
            print(f"Latitude {latitude_input}° not available. Please choose from {self.ZoneLat}")
            return

        albedo_series = [self.Albedo[step][zone_index] for step in range(len(self.Albedo))]

        plt.figure()
        plt.plot(range(len(albedo_series)), albedo_series, label=f"Albedo at {latitude_input}°", linestyle='-',
                 color='green')
        plt.xlabel("Time Step")
        plt.ylabel("Albedo")
        plt.title(f"Albedo Evolution at Latitude {latitude_input}°")
        plt.legend()
        plt.grid(True)
        plt.show()






class InsolationEBM(EnergyBalanceModel):
    def __init__(self, insolation_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.insolation_data = insolation_data
        self.iterations = len(insolation_data)

    def update_insolation(self, time_index):
        insolation_row = self.insolation_data.iloc[time_index]
        self.R_in = [
            insolation_row['Lat_85'],
            insolation_row['Lat_75'],
            insolation_row['Lat_65'],
            insolation_row['Lat_55'],
            insolation_row['Lat_45'],
            insolation_row['Lat_35'],
            insolation_row['Lat_25'],
            insolation_row['Lat_15'],
            insolation_row['Lat_5']
        ]

    def run_insolation_driven_simulation(self):
        for step in range(self.iterations):
            self.update_insolation(step)
            super().run_simulation()






# Running the insolation model
insolation_model = InsolationEBM(insolation_data)
insolation_model.run_insolation_driven_simulation()
# insolation_model.plot_global_mean_temperature()
# insolation_model.plot_temperature_evolution(15)
insolation_model.plot_albedo_evolution(75)