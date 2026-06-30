# SableVixen’s ML Extension

A MakeCode/Micro:bit machine‑learning extension featuring simple classifiers, neural networks, and sensor‑based data collection.

## Features
- Dataset creation and labeled samples  
- KNN classifier  
- Perceptron (binary classifier)  
- NeuralNet (1 hidden layer)  
- DeepNet (2 hidden layers)  
- Decision‑tree node  
- Accelerometer + sound feature extraction  
- Gesture + keyword detection  
- Model save/load with checksums  

## Installation
Add this extension to your MakeCode Micro:bit project:

https://github.com/SableVixen/SableVixens-ML-extension


## Example
```ts
let ds = ml.createDataset();
ml.recordAccelOnA(ds, "shake");
let knn = ml.createKNN(ds, 3);
let result = ml.knnClassify3(knn, 100, 200, -50);
```

## License
See the LICENSE file.
