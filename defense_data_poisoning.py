"""
Défenses contre data poisoning
"""

defenses = {
    "1. Data Validation": {
        "technique": "Inspecting source + integrity checks",
        "mesures": [
            "Verify data source authenticity",
            "Checksum/SHA validation",
            "Anomaly detection (unusual samples)",
            "Statistical tests (distribution shifts)"
        ]
    },
    
    "2. Robust Training": {
        "technique": "Robust loss functions resistants aux outliers",
        "examples": [
            "Huber loss (vs MSE - less sensitive)",
            "Trimmed mean (remove worst samples)",
            "Certifiable robustness (guarantees)"
        ]
    },
    
    "3. Monitoring": {
        "technique": "Track model behavior over time",
        "signes_alerte": [
            "Sudden accuracy drops",
            "Unexpected behavior on new patterns",
            "Model drifting from baseline",
            "Backdoor triggers detected"
        ]
    }
}

print("[DEFENSE] Data Poisoning Prevention:")
for defense, details in defenses.items():
    print(f"\n{defense}")
