import sys
import argparse
import highway_env

# ====== Ajouter le dossier racine au path ======
# Remplace ce chemin si tu as cloné ton repo ailleurs
PROJECT_PATH = "/content/RL_parking_google_collab"
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

# Maintenant Python peut trouver tes modules
from skills import skill_sets
from configs.config import Config
from training.trainer import Trainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--skill_set", type=str, default="basic")
    args = parser.parse_args()

    # Récupérer les skills choisis
    skills = getattr(skill_sets, args.skill_set)()

    # Créer le Trainer
    trainer = Trainer(
        config=Config,
        skills=skills,
        skill_set_name=args.skill_set
    )

    # Lancer l'entraînement
    trainer.train()
