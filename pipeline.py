import os
import argparse
import subprocess
import sys

SLICES_DIR_DEFAULT = os.environ.get('SLICES_DIR', 'slices')
CFG_OUTPUT_DIR_DEFAULT = os.environ.get('CFG_OUTPUT_DIR', 'cfg_output')
MODELS_DIR_DEFAULT = os.environ.get('MODELS_DIR', 'models')
ORIGINAL_DIR_DEFAULT = '/home/ubuntu/original'


def run(cmd, env=None):
    print("$ " + " ".join(cmd))
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        sys.exit(res.returncode)


def run_slicing(project_root, warnings_file, cfwr_root, slices_dir):
    env = os.environ.copy()
    env['SLICES_DIR'] = os.path.abspath(slices_dir)
    run(['./gradlew', 'run', f"-PappArgs={project_root} {warnings_file} {cfwr_root}"], env=env)


def run_cfg_generation(slices_dir, cfg_output_dir):
    for name in os.listdir(slices_dir):
        if not name.endswith('.java') and not os.path.isdir(os.path.join(slices_dir, name)):
            # Accept either raw .java files or per-slice directories created by Specimin
            continue
        path = os.path.join(slices_dir, name)
        if os.path.isdir(path):
            # Find any .java under this slice directory and generate CFGs per file
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith('.java'):
                        java_file = os.path.join(root, f)
                        base = os.path.splitext(os.path.basename(java_file))[0]
                        out_dir = os.path.join(cfg_output_dir, base)
                        if not os.path.exists(out_dir) or not any(n.endswith('.json') for n in os.listdir(out_dir)):
                            run([sys.executable, 'cfg.py', '--java_file', java_file, '--out_dir', cfg_output_dir])
        else:
            java_file = path
            base = os.path.splitext(os.path.basename(java_file))[0]
            out_dir = os.path.join(cfg_output_dir, base)
            if not os.path.exists(out_dir) or not any(n.endswith('.json') for n in os.listdir(out_dir)):
                run([sys.executable, 'cfg.py', '--java_file', java_file, '--out_dir', cfg_output_dir])


def run_train(model):
    if model == 'hgt' or model == 'all':
        run([sys.executable, 'hgt.py'])
    if model == 'gbt' or model == 'all':
        run([sys.executable, 'gbt.py'])


def run_predict(model, java_file, models_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if model == 'hgt' or model == 'all':
        hgt_model = os.path.join(models_dir, 'best_model.pth')
        hgt_out = os.path.join(out_dir, 'hgt_pred.json')
        run([sys.executable, 'predict_hgt.py', '--java_file', java_file, '--model_path', hgt_model, '--out_path', hgt_out])
    if model == 'gbt' or model == 'all':
        gbt_model = os.path.join(models_dir, 'model_iteration_1.joblib')
        gbt_out = os.path.join(out_dir, 'gbt_pred.json')
        run([sys.executable, 'predict_gbt.py', '--java_file', java_file, '--model_path', gbt_model, '--out_path', gbt_out])


def run_predict_over_original(model, original_root, models_dir, out_root):
    for root, _, files in os.walk(original_root):
        for f in files:
            if f.endswith('.java'):
                java_file = os.path.join(root, f)
                rel = os.path.relpath(java_file, original_root)
                out_dir = os.path.join(out_root, os.path.dirname(rel))
                run_predict(model, java_file, models_dir, out_dir)


def main():
    parser = argparse.ArgumentParser(description='End-to-end pipeline for CFWR')
    parser.add_argument('--steps', default='all', choices=['all','slice','cfg','train','predict','predict-original'], help='Which step to run')
    parser.add_argument('--model', default='all', choices=['all','hgt','gbt'], help='Which model(s) to train/predict')
    parser.add_argument('--slices_dir', default=SLICES_DIR_DEFAULT)
    parser.add_argument('--cfg_output_dir', default=CFG_OUTPUT_DIR_DEFAULT)
    parser.add_argument('--models_dir', default=MODELS_DIR_DEFAULT)
    parser.add_argument('--predict_java_file', help='Slice to predict on when steps include predict')
    parser.add_argument('--predict_out_dir', default='predictions', help='Output directory for predictions')
    parser.add_argument('--project_root', help='Project root for slicing (slice step)')
    parser.add_argument('--warnings_file', help='Warnings file for slicing (slice step)')
    parser.add_argument('--cfwr_root', default=os.getcwd(), help='CFWR root (slice step)')
    parser.add_argument('--original_root', default=ORIGINAL_DIR_DEFAULT, help='Original projects root for bulk prediction')
    args = parser.parse_args()

    os.makedirs(args.slices_dir, exist_ok=True)
    os.makedirs(args.cfg_output_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    if args.steps in ('slice','all'):
        if not args.project_root or not args.warnings_file:
            print('Error: --project_root and --warnings_file are required for slice step')
            sys.exit(2)
        run_slicing(args.project_root, args.warnings_file, args.cfwr_root, args.slices_dir)

    if args.steps in ('cfg','all'):
        run_cfg_generation(args.slices_dir, args.cfg_output_dir)

    if args.steps in ('train','all'):
        run_train(args.model)

    if args.steps == 'predict':
        if not args.predict_java_file:
            print('Error: --predict_java_file is required when running predict step')
            sys.exit(2)
        run_predict(args.model, args.predict_java_file, args.models_dir, args.predict_out_dir)

    if args.steps == 'predict-original':
        run_predict_over_original(args.model, args.original_root, args.models_dir, args.predict_out_dir)


if __name__ == '__main__':
    main()


