# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Score images from model produced by another run."""

import argparse

from azureml.automl.dnn.vision.common import utils
from azureml.automl.dnn.vision.common.logging_utils import get_logger
from azureml.automl.dnn.vision.common.parameters import add_task_agnostic_scoring_parameters
from azureml.automl.dnn.vision.object_detection.common.parameters import add_model_agnostic_od_scoring_parameters
from azureml.automl.dnn.vision.object_detection_yolo.common.constants import YoloLiterals, \
    YoloParameters, inference_settings_defaults
from azureml.automl.dnn.vision.object_detection_yolo.writers.score import score

logger = get_logger(__name__)


TASK_TYPE = "image-object-detection"


@utils._exception_handler
def main() -> None:
    """Wrapper method to execute script only when called and not when imported."""
    parser = argparse.ArgumentParser(description="Object detection scoring (using yolov5)", allow_abbrev=False)
    add_task_agnostic_scoring_parameters(parser, inference_settings_defaults)

    # Model Settings
    # should not set defaults for those model settings arguments to use those from training settings by default
    parser.add_argument(utils._make_arg(YoloLiterals.IMG_SIZE), type=int,
                        help='Image size for inference')

    parser.add_argument(utils._make_arg(YoloLiterals.BOX_SCORE_THRESH), type=float,
                        help="During inference, only return proposals with a score greater than "
                             "box_score_thresh. The score is the multiplication of the objectness score "
                             "and classification probability. (For reproducing validation mAP, "
                             "use {}".format(YoloParameters.DEFAULT_BOX_SCORE_THRESH))

    parser.add_argument(utils._make_arg(YoloLiterals.NMS_IOU_THRESH), type=float,
                        help="IOU threshold used during inference in nms post processing")

    add_model_agnostic_od_scoring_parameters(parser)

    args, unknown = parser.parse_known_args()
    args_dict = vars(args)

    # Set up logging
    task_type = TASK_TYPE
    utils._set_logging_parameters(task_type, args_dict)

    if unknown:
        logger.info("Got unknown args, will ignore them.")

    device = utils._get_default_device()
    settings = utils._merge_dicts_with_not_none_values(args_dict, inference_settings_defaults)

    input_dataset = utils.get_scoring_dataset(dataset_id=args.input_dataset_id,
                                              mltable_json=args.input_mltable_uri)

    score(args.run_id, device, settings=settings,
          experiment_name=args.experiment_name,
          output_file=args.output_file,
          root_dir=args.root_dir,
          image_list_file=args.image_list_file,
          output_dataset_target_path=args.output_dataset_target_path,
          input_dataset=input_dataset,
          validate_score=args.validate_score,
          log_output_file_info=args.log_output_file_info)


if __name__ == "__main__":
    # execute only if run as a script
    main()
