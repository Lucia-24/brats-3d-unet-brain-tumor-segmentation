.PHONY: preprocess patches train plot clean clean-all

preprocess:
	python src/load_data.py

patches:
	python src/build_patches.py

train:
	python src/train_model.py

plot:
	python src/plot_metrics.py

all: preprocess patches train plot

clean:
	rm -f models/*.pt models/*.csv

clean-all:
	rm -rf processed_patients patches_binary models