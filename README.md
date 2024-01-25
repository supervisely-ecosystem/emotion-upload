# Upload pairs of TIFF images
This script converts TIFFs to NRRDs, slice 3D NRRDs to 2D and uploads them to Supervisely.

1. Edit the `local.env` file with your IDs.
2. Execute `sh create_venv.sh` to create a virtual environment and install the dependencies.
3. Edit paths to the `images_data`. The folder must contain two subfolders: `ir_images` and `point_clouds`.
4. Launch the script.