.PHONY: all data etl analysis train predict test clean

all: data etl analysis train

data:      ; python -m src.generate_data
etl:       ; python -m src.etl
analysis:  ; python -m src.analysis
train:     ; python -m src.train_model
predict:   ; python -m src.predict
test:      ; python -m pytest -q

clean:
	rm -f data/raw/*.csv data/processed/*.db models/*.joblib
