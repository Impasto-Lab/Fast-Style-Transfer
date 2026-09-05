import os
from PIL import Image
from torch.utils.data import Dataset

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')


class ImageFolderDataset(Dataset):
    """
    Robust image dataset loader that recursively traverses a folder for all valid image files,
    ignoring subdirectories, hidden files, and non-image artifacts.
    """
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir (str): Root path containing image files (flat or nested subfolders).
            transform (callable, optional): Transform to be applied on a PIL Image.
        """
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: '{root_dir}'")

        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Recursively search for image files
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.startswith('.'):
                    continue
                if fname.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    self.image_paths.append(os.path.join(dirpath, fname))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No valid image files ({', '.join(VALID_IMAGE_EXTENSIONS)}) found in '{root_dir}'")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for occasionally corrupted images during long training runs
            print(f"Warning: Corrupted image encountered at '{img_path}': {e}. Skipping to next.")
            next_idx = (idx + 1) % len(self.image_paths)
            return self.__getitem__(next_idx)

        if self.transform is not None:
            image = self.transform(image)

        return image
