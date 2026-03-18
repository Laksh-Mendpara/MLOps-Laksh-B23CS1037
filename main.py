import os
import argparse
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

from dataset.data_loader import load_and_preprocess_data
from core.tune import train_tune, train_final_model

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Run a quick test with very few samples and epochs.")
    args = parser.parse_args()

    print("Loading data...")
    df, en_vocab, hi_vocab = load_and_preprocess_data(data_path='data/English-Hindi.tsv')
    print(f"Data Loaded: {len(df)} pairs. EN Vocab: {len(en_vocab)}, HI Vocab: {len(hi_vocab)}")

    # Initialize Ray 
    ray.init()

    search_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16, 32, 64]),
        "num_heads": tune.choice([4, 8]),
        "d_model": 512,  # Fixed, must be divisible by num_heads
        "d_ff": tune.choice([1024, 2048]),
        "dropout": tune.uniform(0.1, 0.4),
        "num_epochs": 20  
    }

    num_samples = 20
    if args.test_run:
        print("--- RUNNING QUICK TEST ---")
        num_samples = 4
        search_space["num_epochs"] = 1

    optuna_search = OptunaSearch(metric="bleu", mode="max")
    
    scheduler = ASHAScheduler(
        metric="bleu",
        mode="max",
        max_t=search_space["num_epochs"],
        grace_period=min(5, search_space["num_epochs"]),
        reduction_factor=2
    )

    print("Starting Ray Tune...")
    # Utilize 1 GPU per trial to leverage all 4 GPUs concurrently
    trainable_with_resources = tune.with_resources(
        tune.with_parameters(train_tune, df=df, en_vocab=en_vocab, hi_vocab=hi_vocab),
        resources={"gpu": 1.0}
    )

    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=search_space
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric="bleu", mode="max")
    print("==================================================")
    print("Best trial config:", best_result.config)
    print("Optimization phase complete. Training final model until early stopping is triggered...")
    
    save_path = "b23cs1037_ass_4_best_model.pth"
    
    # Train final model with the best configuration
    final_bleu, final_loss, final_time = train_final_model(
        config=best_result.config,
        df=df, 
        en_vocab=en_vocab, 
        hi_vocab=hi_vocab,
        save_path=save_path
    )

    with open("rollno_ass_4_report.txt", "w") as f:
        f.write("=== BEST MODEL CONFIG ===\n")
        for k, v in best_result.config.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== FINAL RETAINED MODEL METRICS ===\n")
        f.write(f"Loss: {final_loss:.4f}\n")
        f.write(f"BLEU (NLTK): {final_bleu*100:.2f}\n")
        f.write(f"Final Retraining Time: {final_time:.2f} sec\n")

    hf_token = os.getenv("hf_token")
    if hf_token:
        try:
            print("Logging in to Hugging Face...")
            login(token=hf_token)
            api = HfApi()
            username = api.whoami()["name"]
            repo_id = f"{username}/MLOPS-Assignment-4"
            print(f"Pushing to Hugging Face repo: {repo_id}")
            api.create_repo(repo_id=repo_id, exist_ok=True)
            
            api.upload_file(
                path_or_fileobj=save_path,
                path_in_repo=save_path,
                repo_id=repo_id,
                commit_message="Add optimally tuned English-Hindi model (early stopped retrain)"
            )
            print("Successfully pushed to Hugging Face Hub.")
        except Exception as e:
            print(f"Failed to push to Hugging Face: {e}")
    else:
        print("Warning: hf_token not found in Environment. Model was NOT pushed to Hugging Face.")
    
    print("Done!")

if __name__ == "__main__":
    main()
