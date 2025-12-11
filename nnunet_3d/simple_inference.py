import torch
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results


if __name__ == "__main__":
    INPUT_IMAGE = "/path/to/your/image_0000.tiff"
    OUTPUT_FOLDER = "/path/to/your/output_folder"

    # Instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,  # step size for sliding window inference
        use_gaussian=True,
        use_mirroring=False,  # test time augmentation
        perform_everything_on_device=True,
        device=torch.device("cuda", 0),  # use the first GPU, change if necessary
        verbose=False,
        verbose_preprocessing=True,
        allow_tqdm=False,
    )

    # Load the model
    predictor.initialize_from_trained_model_folder(
        join(
            nnUNet_results,
            "path/to/your/trained_model",
        ),
        use_folds="all",  # using all folds for inference, change to specific fold if necessary
        checkpoint_name="checkpoint_final.pth",
    )

    # Run prediction on a single image
    # Input format: list of lists where each inner list contains modalities for one case
    predictor.predict_from_files(
        [[INPUT_IMAGE]],
        OUTPUT_FOLDER,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )
