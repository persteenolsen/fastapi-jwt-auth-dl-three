FEATURES = [
    "Gr_Liv_Area",
    "Overall_Qual",
    "Year_Built",
    "Garage_Cars",
    "Full_Bath",
    "Bedroom_AbvGr",
    "Lot_Area",
    "HouseAge",
    "HasGarage"
]

def transform(data: dict, current_year: int = 2026):
    return {
        "Gr_Liv_Area": data["Gr_Liv_Area"],
        "Overall_Qual": data["Overall_Qual"],
        "Year_Built": data["Year_Built"],
        "Garage_Cars": data["Garage_Cars"],
        "Full_Bath": data["Full_Bath"],
        "Bedroom_AbvGr": data["Bedroom_AbvGr"],
        "Lot_Area": data["Lot_Area"],

        # derived features (IDENTICAL everywhere)
        "HouseAge": current_year - data["Year_Built"],
        "HasGarage": 1 if data["Garage_Cars"] > 0 else 0
    }