import sys
#Need to replace the path to the specific run of ultralytics

sys.path.append('../WeightedBifpn-Sigmoidloss/ultralytics')
from ultralytics import YOLO



def train_evaluate_export_model():
  
    # model = YOLO('yolov8s-obb.yaml')  # build a new model from YAML
    # model = YOLO('yolov8s-obb.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8s-obb.yaml').load('yolov8s-obb.pt')  # build from YAML and transfer weights

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    # You need to replace the the yaml file path
    train_results = model.train(data=r'..\yolov8-obb-basic\datasets\rsdd-sar.yaml', epochs=200, wokers=2, device=0,batch=4,patience=50)
    
    #test

    # Evaluate the model's performance on the validation set
    val_results = model.val()

    # Perform object detection on an image using the model
   

    # Export the model to ONNX format
    export_success = model.export(format='onnx')

    return train_results, val_results, detect_results, export_success

if __name__ == '__main__':
    train_results, val_results, detect_results, export_success = train_evaluate_export_model()