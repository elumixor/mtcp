import autorootcwd  # Do not delete - adds the root of the project to the path

from ml.evaluation import evaluate_rocs, evaluate_confusion_matrices, evaluate_significance, evaluate_feature_importance
from tqdm import tqdm
from ml.nn import Model


def evaluate(
    model: Model,
    roc=True,
    significance=True,
    confusion_matrix=True,
    feature_importance=True,
    val=None,
    batch_size=1024,
    F=None,
    wandb_run=None,
    device="cpu",
    files_dir=None,
):
    model.eval()
    model.to(device)

    if roc:
        evaluate_rocs(
            model,
            val,
            batch_size=batch_size,
            device=device,
            wandb_run=wandb_run,
            files_dir=files_dir,
        )

    thresholds = [None]

    if significance:
        print("Evaluating significance")
        threshold, threshold_simple = evaluate_significance(
            model,
            val,
            F,
            batch_size=batch_size,
            device=device,
            wandb_run=wandb_run,
            files_dir=files_dir,
        )
        thresholds = [None, threshold, threshold_simple]

    if confusion_matrix:
        for threshold in tqdm(thresholds, desc="Evaluating confusion matrices"):
            evaluate_confusion_matrices(
                model,
                val,
                threshold=threshold,
                batch_size=batch_size,
                device=device,
                wandb_run=wandb_run,
                files_dir=files_dir,
            )

    if feature_importance:
        evaluate_feature_importance(model, val, device=device, wandb_run=wandb_run, files_dir=files_dir, num_examples=25)

    return thresholds


if __name__ == "__main__":
    import argparse
    from ml.download_model import download_model
    from ml.nn import ResNet, Transformer
    from ml.utils import get_config
    from ml.data import load_from_config

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Evaluate the trained model")
    parser.add_argument("config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--clean", action="store_true", help="Clean the model directory")
    args = parser.parse_args()

    # Read the config
    model_path = download_model(args.config, clean=args.clean)

    config = get_config(args.config, check_cuda_device=False, silent=True)

    # Load the model
    model_class = ResNet if config.model == "resnet" else Transformer
    model = model_class.from_saved(model_path)

    # Load the validation set
    trn, val, tst = load_from_config(config)

    F = 1 / (1 - config.trn_split)

    thresholds = evaluate(
        model=model,
        val=val,
        batch_size=config.batch_size,
        device=config.device,
        wandb_run=None,
        F=F,
        roc="roc" in config.evaluations,
        significance="significance" in config.evaluations,
        confusion_matrix="confusion_matrix" in config.evaluations,
        feature_importance="feature_importance" in config.evaluations,
        files_dir=f"ml/outputs/plots/{config.run_name}"
    )
