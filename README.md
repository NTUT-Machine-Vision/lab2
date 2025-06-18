# lab2

The code archive for Lab 2 is somewhat messy, as it includes tests for all four models across various platforms and model types.

``` bash
# This is the folder structure of lab02
Lab02/
├── MTK_Inference_result 
├── MTK_dla_weight
├── README.md # This file
├── all raw models 
├── amd_vm_test 
├── t4_vm_test
└── train code 
```

## MTK_Inference_result
This folder contains the inference results of the models. The folder structure is as follows:
``` bash
Lab02/MTK_Inference_result/
├── armnn_benchmark
├── armnn_benchmark.py
├── genio_hardware.jpg
├── neuronrt_benchmark
└── neuronrt_benchmark.py
```
- `armnn_benchmark`: This folder contains the benchmark results of the armnn model.
- `armnn_benchmark.py`: This file contains the code to run the armnn benchmark.
- `genio_hardware.jpg`: This file contains the hardware information of the genio model.
- `neuronrt_benchmark`: This folder contains the benchmark results of the neuronrt model.
- `neuronrt_benchmark.py`: This file contains the code to run the neuronrt benchmark. (the code will convert .tflite to .dla first)

## MTK_dla_weight
This folder contains the weight files of the models. The folder structure is as follows:
``` bash
Lab02/MTK_dla_weight/
├── adamw_float16_mdla3_0.dla
├── adamw_float32_mdla3_0.dla
├── batch_float16_mdla3_0.dla
├── batch_float32_mdla3_0.dla
├── best_float16_mdla3_0.dla
└── best_float32_mdla3_0.dla
```
- `adamw_float16_mdla3_0.dla`: This file contains the weight file of the adamw model in float16 format.
- `adamw_float32_mdla3_0.dla`: This file contains the weight file of the adamw model in float32 format.
- `batch_float16_mdla3_0.dla`: This file contains the weight file of the batch model in float16 format.
- `batch_float32_mdla3_0.dla`: This file contains the weight file of the batch model in float32 format.
- `best_float16_mdla3_0.dla`: This file contains the weight file of the best model in float16 format.
- `best_float32_mdla3_0.dla`: This file contains the weight file of the best model in float32 format.

## all raw models
This folder contains the output of the raw models. The folder structure is as follows:
``` bash
Lab02/all raw models/
├── rtdetr_1000epochs_original
├── yolo11n_1000epochs_adamw
├── yolo11n_1000epochs_batch100
└── yolo11n_1000epochs_original
```
- `rtdetr_1000epochs_original`: This folder contains the raw model of the rtdetr model.
- `yolo11n_1000epochs_adamw`: This folder contains the raw model of the yolo11n model with adamw optimizer.
- `yolo11n_1000epochs_batch100`: This folder contains the raw model of the yolo11n model with batch size 100.
- `yolo11n_1000epochs_original`: This folder contains the raw model of the yolo11n model.

## amd_vm_test
This folder contains the test results of the models on the AMD VM. The folder structure is as follows:
``` bash
Lab02/amd_vm_test/
├── image
├── ntut_vm_client.py
├── output
├── rtdetr_models
└── yolo_models
```
- `image`: This folder contains the images used for testing.
- `ntut_vm_client.py`: This file contains the code to run the test on the AMD VM.
- `output`: This folder contains the output of the test.
- `rtdetr_models`: This folder contains the models of the rtdetr model.
- `yolo_models`: This folder contains the models of the yolo11n model.

## t4_vm_test
This folder contains the test results of the models on the T4 VM. The folder structure is as follows:
``` bash
Lab02/t4_vm_test/
├── inference_results
│   ├── RTDETR
│   └── YOLO
└── inference_script.py
```
- `inference_results`: This folder contains the inference results of the models.
    - `RTDETR`: This folder contains the inference results of the rtdetr model.
    - `YOLO`: This folder contains the inference results of the yolo11n model.
- `inference_script.py`: This file contains the code to run the inference on the T4 VM.

## train code
This folder contains the training code of the models. The folder structure is as follows:
``` bash
Lab02/train code/
└── train_code.md
```
- `train_code.md`: This file contains all the training code of the models, including the training parameters and the path of the models.
