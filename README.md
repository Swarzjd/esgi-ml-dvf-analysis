# Analyse de données immobilières : Machine Learning & Deep Learning

Projet de Machine Learning et de Deep Learning reposant sur l'analyse d'un jeu de données de biens immobiliers.

Lien utile : 
- [Lien vers les données](https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres-geolocalisees/20261020-114721/full-2025.csv.gz)

## Problématique

À partir des caractéristiques d'un bien immobilier, que peut-on en déduire ? Le projet aborde deux tâches complémentaires.

### Classification

À partir des données d'un bien, peut-on le classifier (par exemple par catégorie, type de bien ou gamme de prix) ?

### Régression

À partir des données d'un bien, on cherche à **prédire son prix** en fonction de ses caractéristiques :

- la surface
- le nombre de pièces
- le code postal (`zip code`)

## Structure du projet

```
.
├── data/              # Jeux de données (bruts et nettoyés)
├── src/               # Code source (prétraitement, modèles, évaluation)
├── requirements.txt   # Dépendances Python
├── anaysis-dvf.ipynb  # Notebooks d'exploration et d'expérimentation
└── README.md
```

## Prérequis

- Python 3.9 ou supérieur
- `pip`
- Git (optionnel)

## Installation et lancement

### 1. Cloner le projet

```bash
git clone <url-du-depot>
cd <nom-du-projet>
```

### 2. Créer l'environnement virtuel

```bash
python -m venv .venv
```

### 3. Activer l'environnement virtuel

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows (cmd)**

```cmd
.venv\Scripts\activate.bat
```

Une fois activé, le nom `(.venv)` apparaît au début de votre invite de commande.

### 4. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Lancer le projet

Pour explorer les données et lancer les modèles via Jupyter :

```bash
jupyter notebook
```

ou, si le projet est exécutable en script :

```bash
python src/main.py
```

### 6. Lancer l'application Streamlit

L'application charge les modèles sauvegardés par les notebooks. Exécutez d'abord, jusqu'à leur dernière cellule :

- `analysis-dvf.ipynb` (classification) : crée `models/tree_model.joblib`
- `regression-analysis.ipynb` (régression) : crée `models/regression_model.joblib`

Puis depuis la racine du projet :

```bash
streamlit run src/app.py
```

L'application contient deux pages (menu de gauche) :

- **Appartement ou maison ?** : surface, nombre de pièces et valeur foncière → type de bien prédit
- **Estimation du prix** : type de bien, surface et nombre de pièces → valeur foncière estimée

Sur chaque page, cliquez sur la carte pour placer le bien. Pour l'estimation, le code postal et le département
(attendus par le modèle) sont repris de la transaction connue la plus proche du point cliqué.

### 7. Désactiver l'environnement

```bash
deactivate
```
## Méthodologie

1. **Exploration des données (EDA)** : statistiques descriptives, valeurs manquantes, visualisations
2. **Prétraitement** : nettoyage, encodage du code postal, normalisation
3. **Classification** : entraînement et comparaison de modèles (régression logistique, Random Forest, réseau de neurones...)
4. **Régression** : prédiction du prix (régression linéaire, Random Forest, réseau de neurones...)
5. **Évaluation** : accuracy, F1-score pour la classification ; RMSE, MAE, R² pour la régression