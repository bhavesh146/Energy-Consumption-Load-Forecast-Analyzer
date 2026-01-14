import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load & Preprocess Data
# -----------------------------
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    df['DateTime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('DateTime')

    df['Hour'] = df['DateTime'].dt.hour
    df['Weekday'] = df['DateTime'].dt.weekday
    df['Day_Type'] = np.where(df['Weekday'] < 5, 'Weekday', 'Weekend')

    df['Total_Load'] = (
        df['PowerConsumption_Zone1'] +
        df['PowerConsumption_Zone2'] +
        df['PowerConsumption_Zone3']
    )

    df['Peak_Type'] = np.where(df['Hour'].between(18, 22), 'Peak', 'Off-Peak')

    return df


# -----------------------------
# Peak vs Off-Peak
# -----------------------------
def peak_offpeak_analysis(df):
    result = df.groupby('Peak_Type')['Total_Load'].mean()
    print("\nAverage Load (Peak vs Off-Peak)")
    print(result)


# -----------------------------
# Wastage Detection
# -----------------------------
def wastage_detection(df):
    mean_load = df['Total_Load'].mean()
    std_load = df['Total_Load'].std()

    df['Wastage_Anomaly'] = df['Total_Load'] > (mean_load + 2 * std_load)
    print("\nTotal Wastage Events:", df['Wastage_Anomaly'].sum())

    return mean_load, std_load


# -----------------------------
# Rolling Trend
# -----------------------------
def rolling_trend(df):
    df['Rolling_7'] = df['Total_Load'].rolling(7).mean()
    df['Rolling_30'] = df['Total_Load'].rolling(30).mean()

    plt.figure()
    plt.plot(df['DateTime'], df['Total_Load'], label="Actual")
    plt.plot(df['DateTime'], df['Rolling_7'], label="7-Day Avg")
    plt.plot(df['DateTime'], df['Rolling_30'], label="30-Day Avg")
    plt.legend()
    plt.grid()
    plt.title("Rolling Load Trend")
    plt.show()


# -----------------------------
# Weekday vs Weekend
# -----------------------------
def weekday_weekend(df):
    result = df.groupby('Day_Type')['Total_Load'].mean()
    print("\nWeekday vs Weekend Load")
    print(result)

    result.plot(kind='bar')
    plt.title("Weekday vs Weekend Load")
    plt.ylabel("Average Load (kW)")
    plt.grid()
    plt.show()


# -----------------------------
# Peak Load Risk
# -----------------------------
def peak_load_risk(df):
    hourly = df.groupby('Hour')['Total_Load'].mean()
    threshold = hourly.mean() + hourly.std()

    high_risk = hourly[hourly > threshold]
    print("\nHigh Risk Peak Hours:")
    print(high_risk)

    high_risk.plot(kind='bar')
    plt.title("High Risk Peak Load Hours")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Load")
    plt.grid()
    plt.show()

    return high_risk


# -----------------------------
# Zone Efficiency
# -----------------------------

def zone_efficiency(df):
    zone_avg = {
        'Zone1': df['PowerConsumption_Zone1'].mean(),
        'Zone2': df['PowerConsumption_Zone2'].mean(),
        'Zone3': df['PowerConsumption_Zone3'].mean()
    }

    max_load = max(zone_avg.values())

    zones = []
    scores = []

    print("\nZone Efficiency Scores:")
    for z, v in zone_avg.items():
        score = 100 - (v / max_load) * 40
        score = round(score, 2)
        print(z, ":", score)
        zones.append(z)
        scores.append(score)

    # matplotlib initialization & plot
    plt.figure()
    plt.bar(zones, scores)
    plt.xlabel("Zones")
    plt.ylabel("Efficiency Score")
    plt.title("Zone Efficiency Comparison")
    plt.show()



# -----------------------------
# Statistical Anomaly
# -----------------------------
def statistical_anomaly(df, mean_load, std_load):
    df['Z_Score'] = (df['Total_Load'] - mean_load) / std_load
    df['Stat_Anomaly'] = df['Z_Score'].abs() > 2.5

    print("Statistical Anomalies:", df['Stat_Anomaly'].sum())

    plt.figure()
    plt.plot(df['DateTime'], df['Total_Load'])
    plt.scatter(
        df[df['Stat_Anomaly']]['DateTime'],
        df[df['Stat_Anomaly']]['Total_Load'],
        color='red',
        label= 'Anomaly'
    )
    plt.title("Statistical Anomaly Detection (Power Wastage)")
    plt.xlabel("DateTime")
    plt.ylabel("Load (kW)")
    plt.legend()
    plt.grid()
    plt.show()


# -----------------------------
# Demand Balancing
# -----------------------------
def demand_balancing(df, high_risk):
    peak = df[df['Peak_Type'] == 'Peak']['Total_Load'].mean()
    off_peak = df[df['Peak_Type'] == 'Off-Peak']['Total_Load'].mean()

    print("\nDemand Balancing Suggestions:")
    if peak > off_peak * 1.15:
        print("✔ Shift load to off-peak hours")
    if len(high_risk) > 4:
        print("✔ Stagger usage during peak hours")


# =============================
# MAIN MENU (WHILE LOOP)
# =============================
df = load_and_preprocess("powerconsumption.csv")
mean_load, std_load = wastage_detection(df)
high_risk_hours = None

while True:
    print("\n🔋 SMART ENERGY ANALYTICS SYSTEM")
    print("1. Peak vs Off-Peak Analysis")
    print("2. Rolling Average Trend")
    print("3. Weekday vs Weekend Analysis")
    print("4. Peak Load Risk Identification")
    print("5. Zone Efficiency Scoring")
    print("6. Statistical Anomaly Detection")
    print("7. Demand Balancing Strategy")
    print("8. Exit")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        peak_offpeak_analysis(df)

    elif choice == 2:
        rolling_trend(df)

    elif choice == 3:
        weekday_weekend(df)

    elif choice == 4:
        high_risk_hours = peak_load_risk(df)

    elif choice == 5:
        zone_efficiency(df)

    elif choice == 6:
        statistical_anomaly(df, mean_load, std_load)

    elif choice == 7:
        if high_risk_hours is not None:
            demand_balancing(df, high_risk_hours)
        else:
            print("⚠ Run Peak Load Risk first")

    elif choice == 8:
        print("Exiting System... Thank You!")
        break

    else:
        print("❌ Invalid choice. Try again.")
