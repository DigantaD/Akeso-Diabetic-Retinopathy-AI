from vision_etl.s3_loader import load_csv_from_s3

class GradingLoader:
    def __init__(self, label_csv_s3_key):
        """
        Args:
            label_csv_s3_key (str): S3 key to the grading CSV
        """
        self.label_df = load_csv_from_s3(label_csv_s3_key)

        # Clean up column names
        self.label_df.columns = [col.strip() for col in self.label_df.columns]

        # Ensure image names are clean
        self.label_df["Image name"] = self.label_df["Image name"].astype(str).str.strip()

        # Build a mapping: image_id → grade
        self.label_map = {
            row["Image name"]: int(row["Retinopathy grade"])
            for _, row in self.label_df.iterrows()
        }

    def get_label(self, image_id):
        """
        Args:
            image_id (str): Image file name without path or extension
        Returns:
            int or None: Grade label
        """
        return self.label_map.get(image_id)

    def get_all_image_ids(self):
        """
        Returns:
            List[str]: All image IDs present in the grading CSV
        """
        return list(self.label_map.keys())