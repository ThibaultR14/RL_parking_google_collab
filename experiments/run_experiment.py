import sys
import argparse
import highway_env

# ====== Ajouter le dossier racine au path ======
PROJECT_PATH = "/content/RL_parking_google_collab"  # adapte selon ton environnement
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

# Maintenant Python peut trouver tes modules
from skills import skill_sets
from configs.config import Config
from training.trainer import Trainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--skill_set", type=str, default="basic")
    parser.add_argument("--algo", type=str, default="dqn", help="Choisir 'dqn' ou 'td3'")
    args = parser.parse_args()

    # ===== Récupérer les skills uniquement pour DQN =====
    skills = getattr(skill_sets, args.skill_set)() if args.algo.lower() == "dqn" else None

    # ===== Créer le Trainer =====
    trainer = Trainer(
        config=Config,
        skills=skills,
        skill_set_name=args.skill_set if args.algo.lower() == "dqn" else None,
        algo=args.algo
    )

    # ===== Lancer l'entraînement =====
    trainer.train()