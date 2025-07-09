import torch
import os

def federated_weight_transfer(
    grading_ckpt,
    segmentation_ckpt,
    localization_ckpt,
    router_ckpt,
    verbose=True
):
    print("[ROUTER] 🧠 [Federation] Loading router checkpoint...")

    try:
        router_state = torch.load(router_ckpt)
        router_state = router_state["router_state_dict"] if isinstance(router_state, dict) and "router_state_dict" in router_state else router_state
    except (FileNotFoundError, KeyError) as e:
        print(f"[ROUTER] ❌ Failed to load router weights: {e}")
        return

    all_router_keys = list(router_state.keys())
    # print(f"[ROUTER] 📋 Keys in router_state: {all_router_keys}")

    # Normalize: handle both flat and prefixed keys
    router_weights = {}
    for k, v in router_state.items():
        router_weights[k] = v
        if not k.startswith("router."):
            router_weights[f"router.{k}"] = v
        else:
            unprefixed = k.replace("router.", "")
            router_weights[unprefixed] = v

    if not router_weights:
        print("[ROUTER] ⚠️ No usable keys found in router checkpoint. Aborting.")
        return

    print(f"[ROUTER] 🔍 Total router keys available for transfer: {len(router_weights)}")

    summary = {}

    for ckpt_path in [grading_ckpt, segmentation_ckpt, localization_ckpt]:
        model_name = os.path.basename(ckpt_path)
        print(f"[ROUTER] 🔄 Updating {model_name}...")

        try:
            state = torch.load(ckpt_path)
        except Exception as e:
            print(f"[ROUTER] ❌ Failed to load checkpoint {ckpt_path}: {e}")
            summary[model_name] = "Failed to load"
            continue

        model_state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        # print(f"[ROUTER] 🔎 Keys in {model_name}: {list(model_state.keys())[:20]}... Total: {len(model_state)}")

        matched_keys = 0
        for rk, v in router_weights.items():
            if rk in model_state:
                model_state[rk] = v
                matched_keys += 1
                if verbose:
                    print(f"[ROUTER] ✅ Updated: {rk}")
            elif verbose:
                print(f"[ROUTER] ⚠️ Skipped: {rk} (not found)")

        if matched_keys > 0:
            torch.save({"model_state_dict": model_state}, ckpt_path)
            print(f"[ROUTER] 💾 Saved updated {model_name} with {matched_keys} router weights.")
            summary[model_name] = f"Updated {matched_keys} keys"
        else:
            print(f"[ROUTER] ❌ No router keys matched for {model_name}. Skipping save.")
            summary[model_name] = "No keys matched"

    print("[ROUTER] ✅ [Federation] Weight transfer complete!")
    print(f"[ROUTER] 📊 Summary: {summary}")
    return summary