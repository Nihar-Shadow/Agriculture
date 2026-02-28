import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==============================
# FILE PATHS
# ==============================
CROP_DATA_PATH = "Crop_recommendation.csv"
FERT_DATA_PATH = "fertilizer.csv"
MODEL_PATH = "crop_model.pkl"

# ==============================
# TRAIN OR LOAD MODEL
# ==============================
def train_or_load_model():
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded")
        return model
    except:
        print("⚠ Training model first time...")

        df = pd.read_csv(CROP_DATA_PATH)

        X = df[['N','P','K','ph','rainfall']]
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_PATH)

        print("✅ Model trained and saved")
        return model

# ==============================
# CROP PREDICTION
# ==============================
def crop_recommendation(model):
    print("\n=== Crop Recommendation ===")

    N = float(input("Nitrogen: "))
    P = float(input("Phosphorous: "))
    K = float(input("Potassium: "))
    ph = float(input("pH: "))
    rainfall = float(input("Rainfall (mm): "))

    data = [[N,P,K,ph,rainfall]]

    pred = model.predict(data)[0]

    print(f"\n🌱 Recommended Crop: {pred.upper()}")

# ==============================
# FERTILIZER ADVICE
# ==============================
def fertilizer_advice():
    print("\n=== Fertilizer Advice ===")

    fert_df = pd.read_csv(FERT_DATA_PATH)

    crop = input("Enter crop name: ").lower()

    row = fert_df[fert_df['Crop'].str.lower() == crop]

    if row.empty:
        print("❌ Crop not found in fertilizer database")
        return

    ideal_N = float(row.iloc[0]['N'])
    ideal_P = float(row.iloc[0]['P'])
    ideal_K = float(row.iloc[0]['K'])

    soil_N = float(input("Your Soil Nitrogen: "))
    soil_P = float(input("Your Soil Phosphorous: "))
    soil_K = float(input("Your Soil Potassium: "))

    print("\n📊 Nutrient Analysis & Fertilizer Recommendation:")

    def recommend(name, soil, ideal, fert_name, fert_percent):
        diff = ideal - soil

        if diff <= 0:
            print(f"{name}: OPTIMAL")
        else:
            qty = round(diff / fert_percent * 100, 2)
            print(f"{name}: LOW")
            print(f"➡ Use {fert_name}")
            print(f"➡ Approx requirement: {qty} kg per acre\n")

    recommend("Nitrogen", soil_N, ideal_N, "Urea", 46)
    recommend("Phosphorous", soil_P, ideal_P, "DAP", 46)
    recommend("Potassium", soil_K, ideal_K, "MOP (Potash)", 60)


# ==============================
# MAIN MENU
# ==============================
def main():
    model = train_or_load_model()

    while True:
        print("\n====== FARM AI TERMINAL ======")
        print("1. Crop Recommendation")
        print("2. Fertilizer Advice")
        print("3. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            crop_recommendation(model)
        elif choice == "2":
            fertilizer_advice()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice")

# ==============================
if __name__ == "__main__":
    main()
