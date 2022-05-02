import csv
import matplotlib.pyplot as plot

temps = []
res = []

with open('res.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        temps.append(float(row['Temp(V)']))
        res.append(float(row['Res in miliohms-cm']))

plot.title("YBCO Resistivity")
plot.xlabel("Temp (K)")
plot.ylabel("Resistivity (mOhm cm)")

plot.plot(temps, res, color='blue')
plot.show()