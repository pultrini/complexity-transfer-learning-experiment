import shutil
from pathlib import Path

RAW_TRAIN_DIR = Path("data/CatVsDogs_raw")  # ajuste para onde você extraiu o zip
OUT_DIR = Path("data/CatsVsDogs/train")

def main() -> None:
    (OUT_DIR / "cat").mkdir(parents=True, exist_ok=True)
    (OUT_DIR/ "dog").mkdir(parents=True, exist_ok=True)
    
    
    for img_path in RAW_TRAIN_DIR.glob("*.jpg"):
        label = "cat" if img_path.name.startswith("Cat") else "Dog"
        shutil.copy(img_path, OUT_DIR / label / img_path.name)
        
    
    n_cats = len(list((OUT_DIR/"cat").glob("*.jpg")))
    n_dogs = len(list((OUT_DIR/"cat").glob("*.jpg")))
    print(f"Done. cats: {n_cats}, dogs: {n_dogs}")
    
if __name__ == "__main__":
    main()