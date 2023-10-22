# Quick Start
Follow these steps to quickly get started with using the pre-trained model for document forgery detection:

<a href="https://drive.google.com/drive/folders/1yvt2o_aGprG-3JrcrrxtkPbGBI3rrMQ4?usp=sharing">Google Drive Link</a> and download the model.tar file

Download Pre-trained Model: Download the pre-trained frozen graph from the link provided above.

Load the Model: In your Python script or Jupyter Notebook, load the pre-trained model using TensorFlow's model loading functions. Replace 'path/to/pretrained/model' with the actual file path to the downloaded model:
```bash
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG']) #pipeline , included with the model
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
```

# Getting Output
```
img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.show()
```
## NOTE create label map beforehand, eg
```bash
item {
name : "category1",
id : 1
}
```
# Alternate Way
Run the commands in object_detection.ipynb file included in the google drive
