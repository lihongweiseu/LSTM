# Codes for the paper "A hybrid structural modeling approach leveraging long short-term memory neural network model and physical model"
---  Coded in the MATLAB Simulink environment and Python environment utilizing the powerful deep learning library PyTorch.
## Numerical examples
### Numerical examples in Section 2 (device modeling)
1. _BLWN_training_data_prep.m_ is the MATLAB code file to generate the training-validation-testing data.
2. _BLWN_training_data.mat_ is the training data saved using the MATLAB code file _BLWN_training_data_prep.m_.
3. _BLWN_training.py_ is the Python code file to train LSTM models for representing the MBW and MFD models.
4. _BLWN_MBW_trained_model.pt_ and _BLWN_MDZ_trained_model.pt_ are the saved models for representing the MBW and MFD models after training 20,000 times.

### Numerical examples in Section 3 (substructuring modeling)
1. _MBW.onnx_ and _MFD.onnx_ are the ONNX models of _BLWN_MBW_trained_model.pt_ and _BLWN_MDZ_trained_model.pt_.
2. _ONNX_to_Simulink.m_ is the MATLAB code file to create the Simulink LSTM models _MBW_Simulink.mat_ and _MFD_Simulink.mat_ based on _MBW.onnx_ and _MFD.onnx_.
3. all _.txt_ files are ground acceleration files.
4. _Inter_story_damped_structure_MBW_main.m_ is the main MATLAB code file to simulate the numerical example of the inter-story-damped structure installed with the MBW dampers.
5. _Inter_story_damped_structure_MBW_reference.slx_ is the Simulink model to calculate the reference responses of the inter-story-damped structure installed with the MBW dampers.
6. _Inter_story_damped_structure_MBW_LSTM.slx_ is the Simulink model to calculate the LSTM-based prediction responses of the inter-story-damped structure installed with the MBW dampers.
7. _Inter_story_damped_structure_MFD_main.m_ is the main MATLAB code file to simulate the numerical example of the inter-story-damped structure installed with the MFD dampers.
5. _Inter_story_damped_structure_MFD_reference.slx_ is the Simulink model to calculate the reference responses of the inter-story-damped structure installed with the MFD dampers.
6. _Inter_story_damped_structure_MFD_LSTM.slx_ is the Simulink model to calculate the LSTM-based prediction responses of the inter-story-damped structure installed with the MFD dampers.
7. _Base_isolated_structure_main.m_ is the main MATLAB code file to simulate the numerical example of the based-isolated structure.
8. _Base_isolated_structure_infor.m_ and _Base_isolated_structure_plan.m_ are MATLAB code files to generate the structural parameters and base information for the based-isolated structure.
9. _Base_isolated_structure_refence.slx_ is the Simulink model to calculate the reference responses of the based-isolated structure.
10. _Base_isolated_structure_LSTM.slx_ is the Simulink model to calculate the LSTM-based prediction responses of the based-isolated structure.

## Experimental examples (MR damper modeling)
1. _MR_training_data_prep.m_ is the MATLAB code file to generate the training-testing data.
2. _MR_training_data.mat_ is the training data saved using the MATLAB code file _MR_training_data_prep.m_.
3. _MR_training.py_ is the Python code file to train LSTM models to fit the experimental data of the MR damper.
4. _MR_trained_model.pt_ is the saved model to fit the experimental data of the MR damper after training 10,000 times.
5. _MR.onnx_ is the ONNX model of _MR_trained_model.pt_.
6. _MR_compare.m_ is the MATLAB code file to compare the LSTM model with the Narx model.
