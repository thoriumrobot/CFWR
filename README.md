Currently, this project reads the warnings from Checker Framework and calls Specimin on the right field or method.

Setup:

1) Initialize submodules
   git submodule update --init --recursive

2) Download Checker Framework and set classpath (Lower Bound / Index checker)
   export CHECKERFRAMEWORK_CP="/absolute/path/to/checker-framework/checker/dist/checker-qual.jar:/absolute/path/to/checker-framework/checker/dist/checker.jar"

3) Python deps
   pip install -r requirements.txt

4) Run CFWR

Usage:

Replace ${...} with paths.

project-root=...
warning-log-file=...
CFWR-root=...

./gradlew run -PappArgs="${project-root} ${warning-log-file} ${CFWR-root}"

OR

mvn clean compile exec:java -Dexec.args="${project-root} ${warning-log-file} ${CFWR-root}"

Example:

./gradlew run -PappArgs="/home/ubuntu/naenv/checker-framework/checker/tests/index/ /home/ubuntu/naenv/checker-framework/index1.out /home/ubuntu/naenv/CFWR/"

OR

mvn clean compile exec:java -Dexec.args="/home/ubuntu/naenv/index_sub/ /home/ubuntu/naenv/checker-framework/index1.out /home/ubuntu/naenv/CFWR/"

Generating CFGs for slices and training:

- Place Specimin slices (.java) under the directory referenced by SLICES_DIR (default: slices/).
- Generate CFGs and train models (HGT and GBT) using environment variables to configure paths:

  export SLICES_DIR="/abs/path/to/slices"
  export CFG_OUTPUT_DIR="/abs/path/to/cfg_output"
  export MODELS_DIR="/abs/path/to/models"
  export CHECKERFRAMEWORK_CP="/abs/path/to/checker-framework/checker/dist/checker-qual.jar:/abs/path/to/checker-framework/checker/dist/checker.jar"

  python hgt.py
  python gbt.py

Predictions:

- HGT model
  python predict_hgt.py --java_file /abs/path/to/slice.java \
                        --model_path /abs/path/to/models/best_model.pth \
                        --out_path /abs/path/to/output/hgt_pred.json

- GBT model
  python predict_gbt.py --java_file /abs/path/to/slice.java \
                        --model_path /abs/path/to/models/model_iteration_1.joblib \
                        --out_path /abs/path/to/output/gbt_pred.json

End-to-end pipeline (manual control via flags):

  # Generate CFGs only
  python pipeline.py --steps cfg --slices_dir "$SLICES_DIR" --cfg_output_dir "$CFG_OUTPUT_DIR"

  # Train both models (assumes CFGs exist)
  python pipeline.py --steps train --model all

  # Predict with both models for a specific slice
  python pipeline.py --steps predict --model all \
                     --predict_java_file /abs/path/to/slice.java \
                     --models_dir "$MODELS_DIR" \
                     --cfg_output_dir "$CFG_OUTPUT_DIR" \
                     --predict_out_dir ./predictions

  # Run slicing with CheckerFrameworkWarningResolver and save slices under SLICES_DIR
  export SLICES_DIR=/abs/path/to/slices
  python pipeline.py --steps slice \
      --project_root /home/ubuntu/checker-framework/checker/tests/index/ \
      --warnings_file /home/ubuntu/CFWR/index1.out \
      --cfwr_root /home/ubuntu/CFWR

  # Generate CFGs for all slices and train
  python pipeline.py --steps cfg --slices_dir "$SLICES_DIR" --cfg_output_dir "$CFG_OUTPUT_DIR"
  python pipeline.py --steps train --model all

  # Bulk predictions over original projects
  python pipeline.py --steps predict-original --model all --original_root /home/ubuntu/original --predict_out_dir ./predictions_original
