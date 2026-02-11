import argparse

from skills import skill_sets
from configs.config import Config
from training.trainer import Trainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--skill_set", type=str, default="basic")
    args = parser.parse_args()

    skills = getattr(skill_sets, args.skill_set)()

    trainer = Trainer(
        config=Config,
        skills=skills,
        skill_set_name=args.skill_set
    )

    trainer.train()
