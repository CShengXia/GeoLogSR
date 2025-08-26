import os
import time
import numpy as np
import matplotlib.pyplot as plt
from config import CASCADE_STAGES, CURVE_NAME, DEVICE, BATCH_SIZE
from data_utils import load_data, prepare_datasets_from_multiple_wells, prepare_full_dataset, enhance_high_freq, WellLogDataset
from model import DualChannelSRModel
from train import train_model
from inference import predict, calculate_metrics, save_predictions, save_metrics

def run_super_resolution_stage(stage_config, prev_output_path=None, visualize=False):

    stage_name = stage_config["NAME"]
    print(f"\n{'='*80}\nSTARTING SUPER-RESOLUTION STAGE: {stage_name}\n{'='*80}")
    if prev_output_path is not None and stage_config["TEST_WELL"]["LR_PATH"] == "$PREV_OUTPUT":
        stage_config["TEST_WELL"]["LR_PATH"] = prev_output_path
        print(f"Using previous stage output as input for stage {stage_name}: {prev_output_path}")
    os.makedirs(os.path.dirname(stage_config["OUTPUT_PATH"]), exist_ok=True)
    os.makedirs(os.path.dirname(stage_config["METRICS_PATH"]), exist_ok=True)
    window_size = 64
    X_train, Y_train = prepare_datasets_from_multiple_wells(stage_config["TRAIN_WELLS"], window_size=window_size, stride=16)
    test_lr_x, test_lr_y = load_data(stage_config["TEST_WELL"]["LR_PATH"])
    test_hr_x, test_hr_y = load_data(stage_config["TEST_WELL"]["HR_PATH"])
    print(f"Loaded test data for stage {stage_name}: {len(test_lr_y)} low-res points, {len(test_hr_y)} high-res points.")
    if X_train.size == 0:
        print(f"Error: No training samples for stage {stage_name}. Skipping stage.")
        return None
    num_samples = len(X_train)
    train_size = int(0.8 * num_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    train_X, train_Y = X_train[train_idx], Y_train[train_idx]
    val_X, val_Y = X_train[val_idx], Y_train[val_idx]
    for i in range(len(train_X)):
        if np.random.rand() < 0.3:  # 30% chance to augment
            train_Y[i] = enhance_high_freq(train_Y[i], factor=np.random.uniform(0.8, 1.5))
    train_X = np.expand_dims(train_X, axis=1).astype(np.float32)
    train_Y = np.expand_dims(train_Y, axis=1).astype(np.float32)
    val_X   = np.expand_dims(val_X, axis=1).astype(np.float32)
    val_Y   = np.expand_dims(val_Y, axis=1).astype(np.float32)
    train_dataset = WellLogDataset(torch.tensor(train_X), torch.tensor(train_Y))
    val_dataset   = WellLogDataset(torch.tensor(val_X), torch.tensor(val_Y))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=min(BATCH_SIZE, len(train_dataset)), shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=min(BATCH_SIZE, len(val_dataset)), shuffle=False)
    model = DualChannelSRModel().to(DEVICE)
    print(f"Initialized model for stage {stage_name} - total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Training model for stage {stage_name}...")
    model = train_model(model, train_loader, val_loader)
    print(f"Training complete for stage {stage_name}.")
    test_X_full = prepare_full_dataset(test_lr_y)
    print("Generating predictions for test data...")
    test_pred = predict(model, test_X_full)
    if len(test_pred) > len(test_hr_x):
        print(f"Truncating prediction from {len(test_pred)} to {len(test_hr_x)} points to match ground truth length.")
        test_pred = test_pred[: len(test_hr_x)]
    elif len(test_pred) < len(test_hr_x):
        print(f"Warning: prediction length {len(test_pred)} is less than ground truth length {len(test_hr_x)}.")
        test_hr_x = test_hr_x[: len(test_pred)]
        test_hr_y = test_hr_y[: len(test_pred)]
    output_path = save_predictions(test_hr_x, test_pred, stage_config["OUTPUT_PATH"])
    print(f"Saved prediction CSV to {output_path}")
    metrics = calculate_metrics(test_hr_y, test_pred)
    print(f"Metrics for {CURVE_NAME} (Stage {stage_name}): MSE={metrics['MSE']:.6f}, MAE={metrics['MAE']:.6f}, RMSE={metrics['RMSE']:.6f}, PCC={metrics['PCC']:.4f}, R2={metrics['R2']:.4f}")
    save_metrics(metrics, stage_config["METRICS_PATH"], CURVE_NAME)
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.plot(test_hr_x, test_hr_y, 'b-', label='Ground Truth (HR)')
        plt.plot(test_hr_x, test_pred, 'r--', label='Predicted')
        plt.title(f"{CURVE_NAME} Stage {stage_name}: Prediction vs Ground Truth")
        plt.xlabel("Depth / Index")
        plt.ylabel("Curve Value")
        plt.legend()
        plt.grid(True)
        results_plot_path = os.path.join(os.path.dirname(stage_config["OUTPUT_PATH"]), f"{CURVE_NAME}_{stage_name}_results.png")
        plt.savefig(results_plot_path)
        plt.close()
        print(f"Saved stage {stage_name} results plot to {results_plot_path}")
    print(f"Stage {stage_name} completed.\n" + "-"*80)
    return output_path

def run_cascaded_super_resolution(visualize=False):

    print("="*80)
    print("CASCADING THROUGH ALL SUPER-RESOLUTION STAGES")
    print(f"Device in use: {DEVICE}")
    print(f"Total stages: {len(CASCADE_STAGES)} (from {CASCADE_STAGES[0]['NAME']} to {CASCADE_STAGES[-1]['NAME']})")
    print("="*80)
    start_time = time.time()
    prev_output = None
    for i, stage_config in enumerate(CASCADE_STAGES, start=1):
        print(f"\n>>> Running stage {i}/{len(CASCADE_STAGES)}: {stage_config['NAME']}")
        stage_output = run_super_resolution_stage(stage_config, prev_output_path=prev_output, visualize=visualize)
        if stage_output is None:
            print(f"Aborting cascade at stage {stage_config['NAME']} due to previous error.")
            return
        prev_output = stage_output
    total_time = time.time() - start_time
    h = int(total_time // 3600); m = int((total_time % 3600) // 60); s = int(total_time % 60)
    print(f"\nAll stages complete. Total execution time: {h}h {m}m {s}s")
    if visualize and prev_output:
        print("Generating final comparison plots...")
        original_lr_x, original_lr_y = load_data(CASCADE_STAGES[0]["TEST_WELL"]["LR_PATH"])
        final_hr_x, final_hr_y = load_data(CASCADE_STAGES[-1]["TEST_WELL"]["HR_PATH"])
        final_pred_x, final_pred_y = load_data(prev_output)
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(original_lr_x, original_lr_y, 'k-')
        plt.title("Original Low-Resolution Input")
        plt.grid(True)
        plt.subplot(3, 1, 2)
        plt.plot(final_hr_x, final_hr_y, 'b-')
        plt.title("Ground Truth High-Resolution")
        plt.grid(True)
        plt.subplot(3, 1, 3)
        plt.plot(final_pred_x, final_pred_y, 'r-')
        plt.title("Final Predicted High-Resolution (Cascaded Output)")
        plt.grid(True)
        plt.tight_layout()
        overview_path = os.path.join(os.path.dirname(prev_output), f"{CURVE_NAME}_cascade_overview.png")
        plt.savefig(overview_path)
        plt.close()
        print(f"Saved cascade overview plot to {overview_path}")
        plt.figure(figsize=(12, 5))
        hr_diff = np.abs(np.diff(final_hr_y, prepend=final_hr_y[0]))
        pred_diff = np.abs(np.diff(final_pred_y, prepend=final_pred_y[0]))
        plt.subplot(2, 1, 1)
        plt.plot(final_hr_x, hr_diff, 'b-')
        plt.title("Ground Truth High-Frequency Components")
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(final_pred_x, pred_diff, 'r-')
        plt.title("Predicted High-Frequency Components")
        plt.grid(True)
        plt.tight_layout()
        hf_path = os.path.join(os.path.dirname(prev_output), f"{CURVE_NAME}_high_freq_comparison.png")
        plt.savefig(hf_path)
        plt.close()
        print(f"Saved high-frequency comparison plot to {hf_path}")