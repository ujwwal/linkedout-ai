import argparse
import os
from pathlib import Path

from app.core.config import settings
from app.services.vector_store import VectorStoreService


def main(csv_path: str):
	service = VectorStoreService()
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV not found at {csv_path}")
	service.load_posts_from_csv(csv_path)
	print(f"FAISS index created and saved to: {settings.FAISS_INDEX_PATH}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Load LinkedIn posts CSV and build FAISS index")
	parser.add_argument(
		"--csv",
		type=str,
		default=str(Path.cwd() / "linkedin_content.csv"),
		help="Path to the linkedin_content.csv file",
	)
	args = parser.parse_args()
	main(args.csv)
