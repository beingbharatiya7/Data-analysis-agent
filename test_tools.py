"""Quick self-test: runs every tool on TWO different datasets (no API key needed)."""
import json
import tools

def show(title, result):
    print(f"\n--- {title}")
    print(json.dumps(result, indent=2, default=str)[:600])

for path, grp, num, num2 in [
    ("sample_data/store_sales.csv", "region", "total_sales", "units"),
    ("sample_data/student_performance.csv", "branch", "exam_score", "study_hours"),
]:
    print("\n" + "=" * 60)
    print(tools.load_dataset(path))
    show("describe_dataset", tools.describe_dataset())
    show("summary_stats", tools.summary_stats(num))
    show("filter_rows", tools.filter_rows(num, ">", "100"))
    show("group_and_aggregate", tools.group_and_aggregate(grp, num, "mean"))
    show("correlation", tools.correlation(num, num2))
    show("plot_chart", tools.plot_chart("bar", grp, num))

print("\n--- error handling (bad column should return a message, not crash)")
print(tools.summary_stats("does_not_exist"))
