import sys
#Need to replace the path to the specific run of ultralytics

sys.path.append('../ultralytics')
from ultralytics import YOLO



def train_evaluate_export_model():
  
    # model = YOLO('yolov8s-obb.yaml')  # build a new model from YAML
    # model = YOLO('yolov8s-obb.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8s-obb.yaml').load('yolov8s-obb.pt')  # build from YAML and transfer weights

    # ------------------------------------------data-augmentation----------------------------------------
    #Offline all R90
    train_results = model.train(data=r'..\datasets\offline-all-R90\rsdd-sar.yaml', epochs=200, wokers=2, device=0,batch=4,patience=50)
    #Offline inshore R90
    train_results = model.train(data=r'..\datasets\offline-inshore-R90\rsdd-sar.yaml', epochs=200, wokers=2, device=0,batch=4,patience=50)
    #Online R90
    train_results = model.train(data=r'..\datasets\rsdd-sar.yaml', epochs=200, wokers=2, device=0,batch=4,patience=50,degrees=90)

    #-------------------------------------------------------------------------------------------------------



    #test
    # train_results = model.train(data=r'E:\yolov8-obb-compare\yolov8-obb-basic\datasets\rsdd-sar.yaml', epochs=35, workers=4,device=0,batch=4,patience=50)

    # Evaluate the model's performance on the validation set
    val_results = model.val()

    # Export the model to ONNX format
    export_success = model.export(format='onnx')

    return train_results, val_results, detect_results, export_success

if __name__ == '__main__':
    train_results, val_results, detect_results, export_success = train_evaluate_export_model()