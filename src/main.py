import os
from typing import List, Tuple

import cv2
import nrrd
import supervisely as sly
from dotenv import load_dotenv

# Ensure that supervisely.env contains SERVER_ADDRESS and API_TOKEN.
load_dotenv(os.path.expanduser("~/supervisely.env"))
# Ensure that local.env contains TEAM_ID and WORKSPACE_ID.
load_dotenv("local.env")

team_id = sly.io.env.team_id()
workspace_id = sly.io.env.workspace_id()

api: sly.Api = sly.Api.from_env()

print(f"API instance created for team_id={team_id}, workspace_id={workspace_id}")

working_dir = os.getcwd()
images_dir = os.path.join(working_dir, "images_data")
temp_dir = os.path.join(working_dir, "temp_data")
sly.fs.mkdir(temp_dir)

project_name = "emotion_upload"
dataset_name = "cameras"


def get_image_pairs(images_dir: str) -> List[Tuple[str, str]]:
    ir_images_dir = os.path.join(images_dir, "ir_images")
    pc_images_dir = os.path.join(images_dir, "point_clouds")
    ir_images = sorted(sly.fs.list_files(ir_images_dir))
    pc_images = sorted(sly.fs.list_files(pc_images_dir))
    assert len(ir_images) == len(pc_images)
    return list(zip(ir_images, pc_images))


def tiff_to_nrrd(tiff_path: str) -> str:
    file_name = sly.fs.get_file_name(tiff_path) + ".nrrd"
    img_np = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
    save_path = os.path.join(temp_dir, file_name)
    nrrd.write(save_path, img_np)
    return save_path


def split_to_channels(nrrd_path: str) -> List[str]:
    image_np = nrrd.read(nrrd_path)[0]
    image_name = sly.fs.get_file_name(nrrd_path)
    image_channels = [image_np[:, :, i] for i in range(image_np.shape[2])]
    channels_paths = []
    for idx, image_channel in enumerate(image_channels):
        save_path = os.path.join(temp_dir, f"{image_name}_channel_{idx}.nrrd")
        nrrd.write(save_path, image_channel)
        channels_paths.append(save_path)
    return channels_paths


if __name__ == "__main__":
    # Retrieving or creating a new project.
    project = api.project.get_or_create(workspace_id, project_name)
    # ! Important. Setting multiview settings for the project.
    api.project.set_multiview_settings(project.id)
    # Creating a new dataset.
    dataset = api.dataset.create(project.id, dataset_name, change_name_if_conflict=True)

    # Retrieving the list of image pairs.
    image_pairs = get_image_pairs(images_dir)

    # Iterating over the image pairs.
    for ir_image, pc_image in image_pairs:
        # Converting the images to nrrd format.
        ir_image_nrrd = tiff_to_nrrd(ir_image)
        pc_image_nrrd = tiff_to_nrrd(pc_image)
        # Getting the image name, which will be used as a group name.
        image_name = sly.fs.get_file_name_with_ext(ir_image_nrrd).split("_")[1]

        # Splitting point cloud image to channels.
        pc_image_nrrd_channels = split_to_channels(pc_image_nrrd)

        # Preparing a list of 2D nrdd image paths for uploading.
        upload_paths = [ir_image_nrrd] + pc_image_nrrd_channels

        # Uploading the images as a group.
        image_infos = api.image.upload_multiview_images(
            dataset.id, image_name, upload_paths
        )
        print(f"Uploaded {image_name} with {len(image_infos)} images")
