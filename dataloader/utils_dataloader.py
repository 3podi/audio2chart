import os

def find_chart_files(root_folder):
    """
    Recursively finds all .chart files in root_folder and its subdirectories.

    Args:
        root_folder (str): Path to the root folder to search.

    Returns:
        list: List of absolute paths to all .chart files found.
    """
    chart_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.chart'):
                chart_files.append(os.path.join(root, file))
    return chart_files