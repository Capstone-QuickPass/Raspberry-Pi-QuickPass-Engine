# Covid App Detection

PreTrained Model: ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
Data Set Labeled with: LabelMG


Train Model

```sh
# navigate to folder
python train.py
# Copy output and run in CMD
```

Run Covid App Detection

```sh
# navigate to folder
python run_covid_app_detection
```

(optional) Tensorboard

```sh
# navigate to  \Tensorflow\workspace\models\CovidAppModel
tensorboard --logdir=train
```


Derived from https://github.com/nicknochnack/RealTimeObjectDetection
