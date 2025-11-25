from metrics import compute_all_metrics, save_metrics_json, save_metrics_csv

def main():
    # 1) Dummy gold answers and predictions
    gold_answers = [
        "Paris is the capital of France",
        "The capital of Germany is Berlin",
        "Tokyo is the capital of Japan",
    ]
    predictions = [
        "paris is the capital of france",    # exact (after normalization)
        "the capital of germany is berlin",  # exact
        "London is the capital of Japan",    # wrong
    ]

    # 2) Fake latencies (ms) for three queries
    latencies_ms = [120.0, 150.0, 200.0]

    # 3) Fake resource + cost numbers
    peak_vram_mb = 8000.0
    params_total_m = 7000.0      # e.g. 7B params
    params_trainable_m = 100.0   # LoRA params
    storage_model_mb = 3500.0
    storage_adapters_mb = 200.0
    storage_index_mb = 500.0
    train_gpu_hours = 2.5

    # 4) Compute metrics
    metrics = compute_all_metrics(
        gold_answers=gold_answers,
        predictions=predictions,
        latencies_ms=latencies_ms,
        peak_vram_mb=peak_vram_mb,
        params_total_m=params_total_m,
        params_trainable_m=params_trainable_m,
        storage_model_mb=storage_model_mb,
        storage_adapters_mb=storage_adapters_mb,
        storage_index_mb=storage_index_mb,
        train_gpu_hours=train_gpu_hours,
        num_gpus=1,
        method_name="dummy_multi_lora",
        dataset_name="dummy_dataset",
        extra_info={"model_id": "dummy-1.0"},
    )

    print("Computed metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 5) Save to JSON and CSV under results/
    save_metrics_json(metrics, "results/test_metrics_dummy.json")
    save_metrics_csv(metrics, "results/test_metrics_dummy.csv")
    print("\nSaved metrics to results/test_metrics_dummy.json and .csv")

if __name__ == "__main__":
    main()
