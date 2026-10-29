import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def main():
    df = pd.read_csv("../data/in/in-filtered_df.csv", encoding="utf-8", dtype={"code_postal": str, "code_departement": str},low_memory=False)

    df = df[df["nature_mutation"] == "Vente"]
    df = df[df["type_local"].isin(["Appartement", "Maison"])]

    # keep only single-row mutations to avoid duplicated prices
    df = df[df.groupby("id_mutation")["id_mutation"].transform("size") == 1]

    df["code_postal"] = df["code_postal"].str.replace(r"\.0$", "", regex=True).str.zfill(5)
    df["code_departement"] = df["code_departement"].str.zfill(2)

    df["date_mutation"] = pd.to_datetime(df["date_mutation"])
    df["year"], df["month"] = df["date_mutation"].dt.year, df["date_mutation"].dt.month

    df = df[(df["surface_reelle_bati"] > 0) & (df["nombre_pieces_principales"] > 0)]
    df = df.dropna(subset=["valeur_fonciere", "longitude", "latitude"])

    cols = ["valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales",
            "code_postal", "code_departement", "type_local", "year", "month",
            "longitude", "latitude"]
    df = df[cols]

    df = pd.get_dummies(df, columns=["code_departement"], dtype=bool)

    df.to_csv("../data/tmp/in-preprocessed_df.csv", index=False)

if __name__ == "__main__":
    logging.info("Début du programme")
    main()

