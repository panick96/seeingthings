from imageai.Classification.Custom import ClassificationModelTrainer
model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsDenseNet121()
model_trainer.setDataDirectory("hoops")
model_trainer.trainModel(num_objects=1, num_experiments=100, enhance_data=True, batch_size=16, show_network_summary=True)