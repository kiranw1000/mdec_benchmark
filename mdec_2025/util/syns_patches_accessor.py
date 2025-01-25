import zipfile
from io import BytesIO

from PIL import Image


class SynsPatchesAccessor:
    def __init__(self, path_syns_patches_zip, split="val"):
        self.path_syns_patches_zip = path_syns_patches_zip
        self.split = split
        self.split_files = self._load_split_files()

    def _load_split_files(self):
        with zipfile.ZipFile(self.path_syns_patches_zip, 'r') as zip_file:
            split_files_path = f"syns_patches/splits/{self.split}_files.txt"
            if split_files_path not in zip_file.namelist():
                raise FileNotFoundError(f"'{split_files_path}' not found in the ZIP archive of the SYNS-Patches dataset.")
            
            split_files_content = zip_file.read(split_files_path).decode("utf-8")
            return split_files_content.splitlines()

    def __len__(self):
        return len(self.split_files)

    def __iter__(self):
        with zipfile.ZipFile(self.path_syns_patches_zip, 'r') as zip_file:
            for line in self.split_files:
                folder_name, image_name = line.split()
                image_path = f"syns_patches/{folder_name}/images/{image_name}"

                if image_path not in zip_file.namelist():
                    raise FileNotFoundError(f"'{image_path}' not found in the ZIP archive of the SYNS-Patches dataset.")

                image_data = zip_file.read(image_path)
                image = Image.open(BytesIO(image_data))
                yield image, f"{folder_name}/{image_name}"
