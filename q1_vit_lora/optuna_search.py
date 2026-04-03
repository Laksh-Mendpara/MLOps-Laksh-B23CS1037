import optuna
import torch
import wandb
from config import Config
from dataset import get_dataloaders
from model import get_model
from train import train_model

def objective(trial):
    config = Config()
    r = trial.suggest_categorical('r', [2, 4, 8, 16])
    alpha = trial.suggest_categorical('alpha', [2, 4, 8, 16])
    epochs = 5
    
    run_name = f"optuna_r{r}_alpha{alpha}"
    wandb.init(
        project=config.WANDB_PROJECT,
        name=run_name,
        group="optuna_search",
        reinit=True,
        config={"r": r, "alpha": alpha, "epochs": epochs}
    )
    
    train_loader, val_loader, _ = get_dataloaders(config)
    model = get_model(config, use_lora=True, r=r, alpha=alpha)
    _, best_val_acc = train_model(model, train_loader, val_loader, config, run_name, epochs=epochs)
    wandb.finish()
    
    return best_val_acc

def run_optuna():
    study = optuna.create_study(direction="maximize", study_name="vit_lora_hpo")
    study.optimize(objective, n_trials=10)
    print("Best trial:", study.best_trial.value)
    return study.best_trial.params

if __name__ == "__main__":
    Config.setup()
    run_optuna()
