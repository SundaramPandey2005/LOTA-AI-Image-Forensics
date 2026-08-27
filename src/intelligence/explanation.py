from typing import Dict, Any


def format_grounded_response(result: Dict[str, Any]) -> str:
    """
    Format structured database evidence into a clear, factual, grounded research answer.
    """
    status = result.get("status")
    if status == "UNSUPPORTED":
        return result.get("message", "Analysis query unsupported.")

    template_id = result.get("template_id")
    data = result.get("data")
    params = result.get("params", {})

    if template_id == "BEST_MODEL_FOR_GENERATOR":
        gen = params.get("generator", "specified generator")
        if not data:
            return f"No recorded experimental evaluations found for generator '{gen}'."
        best = data[0]
        return f"**Top Performing Model on {gen.upper()}**: Experiment `{best['experiment_id']}` ({best['architecture'].upper()}-{best['backbone']}) achieved a score of **{best['score']*100:.2f}%**."

    elif template_id == "HARDEST_GENERATOR":
        if not data:
            return "No cross-generator evaluation records currently found in the database."
        hardest = data[0]
        return f"**Hardest Generator to Detect**: `{hardest['generator_name']}` with an average score of **{hardest['avg_score']*100:.2f}%** (lowest observed: {hardest['min_score']*100:.2f}%)."

    elif template_id == "COMPARE_MODELS":
        comp_list = data.get("comparison", [])
        if not comp_list:
            return "Unable to compare the specified models."
        lines = [f"### Model Comparison: `{data.get('exp_id_a')}` vs `{data.get('exp_id_b')}`\n"]
        lines.append("| Generator | Model A Acc | Model B Acc | Acc Difference |")
        lines.append("| :--- | :---: | :---: | :---: |")
        for item in comp_list:
            a_acc = f"{item['exp_a_acc']*100:.1f}%" if item['exp_a_acc'] is not None else "N/A"
            b_acc = f"{item['exp_b_acc']*100:.1f}%" if item['exp_b_acc'] is not None else "N/A"
            diff = f"{item['diff_acc']*100:+.1f}%" if item['diff_acc'] is not None else "N/A"
            lines.append(f"| {item['generator']} | {a_acc} | {b_acc} | {diff} |")
        return "\n".join(lines)

    elif template_id == "MGPS_EFFECT":
        if not data:
            return "No MGPS ablation experiments found in the database."
        lines = ["### MGPS Patch Selection Ablation Summary\n"]
        for row in data:
            lines.append(f"- **{row['patch_strategy'].capitalize()}**: Average Accuracy = {row['avg_accuracy']*100:.2f}%, AUROC = {row['avg_auroc']*100:.2f}%")
        return "\n".join(lines)

    elif template_id == "LIST_ALL_EXPERIMENTS":
        if not data:
            return "No recorded experiments found in the database."
        lines = ["### Recorded Experiments\n", "| ID | Name | Architecture | Backbone | Normalization |", "| :--- | :--- | :--- | :--- | :--- |"]
        for row in data:
            lines.append(f"| `{row['experiment_id']}` | {row['name']} | {row['architecture']} | {row['backbone']} | {row['normalization']} |")
        return "\n".join(lines)

    else:
        return f"**Analysis Result ({template_id})**: {str(data)}"
