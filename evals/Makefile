# Entry point for the evaluation harness.
#   make -C evals data   # fetch FRLL (CC BY 4.0) and the MediaPipe weights
#   make -C evals evals  # run every arm, write evals/results/
PY ?= ../.venv/bin/python
FIG = https://ndownloader.figshare.com/files
MP  = https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

.PHONY: evals data clean
evals:
	cd .. && $(PY) -m evals.run_all

data: data/frll/london_faces_info.csv data/models/face_landmarker.task pipeline-weights
	@echo "data present"

# Arm 11 runs the pipeline's own detector and pose estimator, so it needs the
# weights `faciometry.models.weights` pins. They live in that module's cache, not
# here, and `analyze` never downloads.
.PHONY: pipeline-weights
pipeline-weights:
	-$(PY) -m faciometry.models.weights fetch yunet sixdrepnet_300wlp

data/frll/london_faces_info.csv:
	mkdir -p data/frll
	curl -sSL -o data/frll/london_faces_info.csv     $(FIG)/27397184
	curl -sSL -o data/frll/neutral_front.zip         $(FIG)/8541961
	curl -sSL -o data/frll/smiling_front.zip         $(FIG)/8541964
	curl -sSL -o data/frll/neutral_left_3quarter.zip $(FIG)/8545675
	curl -sSL -o data/frll/neutral_right_3quarter.zip $(FIG)/8545681
	curl -sSL -o data/frll/neutral_left_profile.zip  $(FIG)/8545678
	curl -sSL -o data/frll/neutral_right_profile.zip $(FIG)/8545684
	curl -sSL -o data/frll/smiling_left_3quarter.zip $(FIG)/8545741

data/models/face_landmarker.task:
	mkdir -p data/models
	curl -sSL -o $@ $(MP)

clean:
	rm -f results/*.json results/*.csv data/frll_mediapipe.npz
