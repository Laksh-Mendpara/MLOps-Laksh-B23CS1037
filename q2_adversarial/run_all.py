import argparse
import csv
import logging
import os

import matplotlib.pyplot as plt
import torch
import wandb
from torch.utils.data import DataLoader

from adversarial_detector import evaluate_detector, generate_adversarial_dataset, train_detector
from config import Config
from dataset import get_dataloaders
from fgsm_art import evaluate_fgsm_art
from fgsm_scratch import evaluate_fgsm_scratch
from logger_utils import setup_logger
from model import get_resnet18, get_resnet34
from train_resnet18 import train_resnet18
from visualize import plot_adversarial_examples


def save_history_csv(config, filename, history):
    if not history:
        return None
    path = os.path.join(config.OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_file_to_wandb(path)
    return path


def save_summary_csv(config, filename, rows):
    if not rows:
        return None
    path = os.path.join(config.OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_file_to_wandb(path)
    return path


def save_file_to_wandb(path):
    if wandb.run is not None and path and os.path.exists(path):
        wandb.save(path)


def log_rows_table(name, rows):
    if wandb.run is None or not rows:
        return
    columns = list(rows[0].keys())
    data = [[row[column] for column in columns] for row in rows]
    wandb.log({name: wandb.Table(columns=columns, data=data)})


def plot_metric_curves(history, config, title, filename_prefix):
    if not history:
        return None

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_loss, marker="o", label="Train")
    axes[0].plot(epochs, val_loss, marker="s", label="Val")
    axes[0].set_title(f"{title}: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="Train")
    axes[1].plot(epochs, val_acc, marker="s", label="Val")
    axes[1].set_title(f"{title}: Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(config.OUTPUT_DIR, f"{filename_prefix}_curves.png")
    plt.savefig(plot_path)
    plt.close(fig)
    save_file_to_wandb(plot_path)
    if wandb.run is not None:
        wandb.log({title: wandb.Image(plot_path)})
    return plot_path


def plot_accuracy_drop(epsilons, acc_scratch, acc_art, config):
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, [a * 100 for a in acc_scratch], "o-", label="FGSM (Scratch)")
    plt.plot(epsilons, [a * 100 for a in acc_art], "s-", label="FGSM (IBM ART)")
    plt.title("Accuracy vs Perturbation Strength (Epsilon)")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(config.OUTPUT_DIR, "accuracy_drop.png")
    plt.savefig(plot_path)
    plt.close()
    save_file_to_wandb(plot_path)
    if wandb.run is not None:
        wandb.log({"Accuracy vs Epsilon Drop": wandb.Image(plot_path)})
    return plot_path


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = {}
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    model.load_state_dict(state_dict)
    model.to(device)
    return metadata


def evaluate_classifier(model, dataloader, config):
    model = model.to(config.DEVICE)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def choose_display_index(epsilons, target_eps):
    return min(range(len(epsilons)), key=lambda idx: abs(epsilons[idx] - target_eps))


def get_or_train_clean_model(train_loader, val_loader, test_loader):
    model_path = os.path.join(Config.OUTPUT_DIR, "resnet18_clean.pth")
    model = get_resnet18(Config.NUM_CLASSES)
    clean_history = []
    loaded_checkpoint = False

    if os.path.exists(model_path):
        try:
            metadata = load_checkpoint(model, model_path, Config.DEVICE)
            clean_history = metadata.get("history", [])
            loaded_checkpoint = True
            logging.info("Loaded existing clean-model checkpoint from %s", model_path)
        except Exception as exc:
            logging.warning("Could not load existing clean-model checkpoint %s: %s", model_path, exc)

    clean_test_acc = None
    if loaded_checkpoint:
        clean_test_acc = evaluate_classifier(model, test_loader, Config)
        logging.info("Existing clean-model checkpoint test accuracy: %.2f%%", clean_test_acc)
        if (
            clean_test_acc < Config.MIN_CLEAN_TEST_ACC
            and Config.RETRAIN_IF_CHECKPOINT_BELOW_TARGET
        ):
            logging.warning(
                "Existing clean-model checkpoint is below assignment target %.2f%% < %.2f%%. Retraining from scratch.",
                clean_test_acc,
                Config.MIN_CLEAN_TEST_ACC,
            )
            loaded_checkpoint = False
            clean_history = []
            model = get_resnet18(Config.NUM_CLASSES)

    if not loaded_checkpoint:
        _, _, clean_history = train_resnet18(model, train_loader, val_loader, Config)
        load_checkpoint(model, model_path, Config.DEVICE)
        clean_test_acc = evaluate_classifier(model, test_loader, Config)
        logging.info("Freshly trained clean-model test accuracy: %.2f%%", clean_test_acc)

    save_history_csv(Config, "clean_resnet18_train_val_table.csv", clean_history)
    plot_metric_curves(clean_history, Config, "Clean ResNet18 Train vs Val", "clean_resnet18_train_val")
    save_file_to_wandb(model_path)
    return model, clean_test_acc, model_path


def run_smoke_test():
    Config.setup()
    Config.validate_runtime()
    setup_logger(Config.OUTPUT_DIR, "q2_adversarial_test.log")
    logging.info("Starting Q2 smoke test...")
    logging.info("Active Device: %s", Config.DEVICE)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(Config)

    model = get_resnet18(Config.NUM_CLASSES).to(Config.DEVICE)
    model.eval()
    inputs, labels = next(iter(test_loader))
    inputs = inputs[:4].to(Config.DEVICE)
    labels = labels[:4].to(Config.DEVICE)

    with torch.no_grad():
        logits = model(inputs)
    assert logits.shape == (4, Config.NUM_CLASSES), f"ResNet18 logits shape mismatch: {logits.shape}"

    smoke_eps = [0.0, Config.EPSILON_VALUES[1]]
    scratch_acc, _ = evaluate_fgsm_scratch(
        model, [(inputs.detach().cpu(), labels.detach().cpu())], Config, smoke_eps
    )
    art_acc, _ = evaluate_fgsm_art(
        model, [(inputs.detach().cpu(), labels.detach().cpu())], Config, smoke_eps
    )
    assert len(scratch_acc) == len(smoke_eps)
    assert len(art_acc) == len(smoke_eps)

    train_dataset, adv_examples = generate_adversarial_dataset(
        model, [(inputs.detach().cpu(), labels.detach().cpu())], Config, attack_type="PGD"
    )
    assert len(train_dataset) == 2 * inputs.shape[0], "Detector dataset size mismatch"

    detector = get_resnet34(num_classes=2).to(Config.DEVICE)
    detector.eval()
    det_inputs, _ = next(iter(DataLoader(train_dataset, batch_size=4, shuffle=False)))
    det_inputs = det_inputs.to(Config.DEVICE)
    with torch.no_grad():
        det_logits = detector(det_inputs)
    assert det_logits.shape[1] == 2, f"Detector logits shape mismatch: {det_logits.shape}"

    logging.info(
        "Q2 smoke test passed: classes=%s fgsm_scratch=%s fgsm_art=%s adv_examples=%s detector_logits=%s",
        len(class_names),
        scratch_acc,
        art_acc,
        len(adv_examples),
        tuple(det_logits.shape),
    )
    print("Q2 --test passed")


def main(test_mode=False):
    if test_mode:
        run_smoke_test()
        return

    Config.setup()
    Config.validate_runtime()
    if Config.DEVICE.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    setup_logger(Config.OUTPUT_DIR, "q2_adversarial.log")
    logging.info("Starting Q2 Adversarial Attack Experiments...")
    logging.info("Active Device: %s", Config.DEVICE)

    assignment_failures = []

    if Config.WANDB_API_KEY:
        wandb.login(key=Config.WANDB_API_KEY)

    wandb.init(
        project=Config.WANDB_PROJECT,
        name="Q2_Adversarial_Attacks",
        group="Q2",
        config={
            "device": Config.DEVICE,
            "batch_size": Config.BATCH_SIZE,
            "epochs": Config.EPOCHS,
            "detector_epochs": Config.DETECTOR_EPOCHS,
            "epsilon_values": Config.EPSILON_VALUES,
            "min_clean_test_acc": Config.MIN_CLEAN_TEST_ACC,
            "min_detector_test_acc": Config.MIN_DETECTOR_TEST_ACC,
        },
    )

    try:
        train_loader, val_loader, test_loader, class_names = get_dataloaders(Config)

        fgsm_rows = []
        detector_rows = []

        print("\n=== Training ResNet18 on Clean CIFAR-10 ===")
        model, clean_test_acc, model_path = get_or_train_clean_model(train_loader, val_loader, test_loader)
        wandb.log({"clean_test_acc": clean_test_acc})

        if clean_test_acc < Config.MIN_CLEAN_TEST_ACC:
            assignment_failures.append(
                f"clean ResNet18 test accuracy {clean_test_acc:.2f}% is below required {Config.MIN_CLEAN_TEST_ACC:.2f}%"
            )

        print("\n=== Running FGSM Attack (Scratch) ===")
        acc_scratch, examples_scratch = evaluate_fgsm_scratch(model, test_loader, Config, Config.EPSILON_VALUES)

        print("\n=== Running FGSM Attack (IBM ART) ===")
        acc_art, examples_art = evaluate_fgsm_art(model, test_loader, Config, Config.EPSILON_VALUES)

        display_idx = choose_display_index(Config.EPSILON_VALUES, Config.DISPLAY_EPSILON)
        if examples_scratch[display_idx]:
            plot_adversarial_examples(
                examples_scratch[display_idx],
                class_names,
                Config,
                f"FGSM Scratch (eps={Config.EPSILON_VALUES[display_idx]})",
                "fgsm_scratch_samples.png",
            )
        if examples_art[display_idx]:
            plot_adversarial_examples(
                examples_art[display_idx],
                class_names,
                Config,
                f"FGSM ART (eps={Config.EPSILON_VALUES[display_idx]})",
                "fgsm_art_samples.png",
            )

        plot_accuracy_drop(Config.EPSILON_VALUES, acc_scratch, acc_art, Config)

        for eps, scratch_acc, art_acc in zip(Config.EPSILON_VALUES, acc_scratch, acc_art):
            fgsm_rows.append(
                {
                    "epsilon": eps,
                    "clean_accuracy": round(clean_test_acc, 4),
                    "fgsm_scratch_accuracy": round(scratch_acc * 100, 4),
                    "fgsm_art_accuracy": round(art_acc * 100, 4),
                    "scratch_drop": round(clean_test_acc - scratch_acc * 100, 4),
                    "art_drop": round(clean_test_acc - art_acc * 100, 4),
                }
            )
        save_summary_csv(Config, "fgsm_accuracy_comparison.csv", fgsm_rows)
        log_rows_table("FGSM Accuracy Comparison", fgsm_rows)

        for attack in ["PGD", "BIM"]:
            print(f"\n=== Training {attack} Adversarial Detector ===")
            train_dataset, _ = generate_adversarial_dataset(model, train_loader, Config, attack_type=attack)
            val_dataset, _ = generate_adversarial_dataset(model, val_loader, Config, attack_type=attack)
            test_dataset, adv_examples = generate_adversarial_dataset(model, test_loader, Config, attack_type=attack)

            train_l = DataLoader(
                train_dataset,
                batch_size=Config.DETECTOR_BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.DEVICE.startswith("cuda"),
                persistent_workers=Config.NUM_WORKERS > 0,
            )
            val_l = DataLoader(
                val_dataset,
                batch_size=Config.DETECTOR_BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.DEVICE.startswith("cuda"),
                persistent_workers=Config.NUM_WORKERS > 0,
            )
            test_l = DataLoader(
                test_dataset,
                batch_size=Config.DETECTOR_BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.DEVICE.startswith("cuda"),
                persistent_workers=Config.NUM_WORKERS > 0,
            )

            if adv_examples:
                plot_adversarial_examples(
                    adv_examples,
                    class_names=class_names,
                    config=Config,
                    title=f"{attack} Adversarial Samples",
                    filename=f"{attack.lower()}_samples.png",
                )

            detector_model, best_val_acc, detector_history = train_detector(
                get_resnet34(num_classes=2), train_l, val_l, Config, attack
            )
            det_test_acc = evaluate_detector(detector_model, test_l, Config)
            detector_path = os.path.join(Config.OUTPUT_DIR, f"detector_{attack}.pth")
            save_file_to_wandb(detector_path)

            wandb.log(
                {
                    f"Final_{attack}_Detector_Val_Acc": best_val_acc,
                    f"Final_{attack}_Detector_Test_Acc": det_test_acc,
                }
            )
            save_history_csv(Config, f"{attack.lower()}_detector_train_val_table.csv", detector_history)
            plot_metric_curves(
                detector_history,
                Config,
                f"{attack} Detector Train vs Val",
                f"{attack.lower()}_detector_train_val",
            )

            if det_test_acc < Config.MIN_DETECTOR_TEST_ACC:
                assignment_failures.append(
                    f"{attack} detector test accuracy {det_test_acc:.2f}% is below required {Config.MIN_DETECTOR_TEST_ACC:.2f}%"
                )

            detector_rows.append(
                {
                    "attack": attack,
                    "best_val_acc": round(best_val_acc, 4),
                    "test_acc": round(det_test_acc, 4),
                    "meets_target": det_test_acc >= Config.MIN_DETECTOR_TEST_ACC,
                }
            )

        save_summary_csv(Config, "detector_accuracy_comparison.csv", detector_rows)
        log_rows_table("Detector Accuracy Comparison", detector_rows)
        save_file_to_wandb(model_path)
        save_file_to_wandb(os.path.join(Config.OUTPUT_DIR, "q2_adversarial.log"))
    finally:
        wandb.finish()

    if assignment_failures:
        raise RuntimeError(
            "Q2 assignment targets were not met:\n- " + "\n- ".join(assignment_failures)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Q2 adversarial attack experiments.")
    parser.add_argument("--test", action="store_true", help="Run a lightweight smoke test for Q2.")
    args = parser.parse_args()
    main(test_mode=args.test)
