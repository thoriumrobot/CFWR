#!/usr/bin/env python3
"""
CFWR Pipeline Script with Best Practices Defaults

This script orchestrates the complete CFWR pipeline using:
- Dataflow-augmented CFGs by default
- Augmented slices as default behavior
- Checker Framework slicer as default
- Consistent integration across all components

Best Practices:
- Uses dataflow information in CFGs for better model performance
- Prefers augmented slices over original slices for training
- Uses Checker Framework slicer for better slice quality
- Maintains consistency between training and prediction pipelines
"""

import os
import argparse
import subprocess
import sys

# Best practices defaults
SLICES_DIR_DEFAULT = os.environ.get('SLICES_DIR', 'slices')
CFG_OUTPUT_DIR_DEFAULT = os.environ.get('CFG_OUTPUT_DIR', 'cfg_output')
MODELS_DIR_DEFAULT = os.environ.get('MODELS_DIR', 'models')
ORIGINAL_DIR_DEFAULT = '/home/ubuntu/original'
AUGMENTED_DIR_DEFAULT = os.environ.get('AUGMENTED_SLICES_DIR', 'slices_aug')


def run(cmd, env=None):
    print("$ " + " ".join(cmd))
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        sys.exit(res.returncode)


def run_slicing(project_root, warnings_file, cfwr_root, base_slices_dir, slicer_type='cf'):
    # Create slicer-specific directory
    slices_dir = os.path.join(base_slices_dir, f"slices_{slicer_type}")
    os.makedirs(slices_dir, exist_ok=True)
    
    env = os.environ.copy()
    env['SLICES_DIR'] = os.path.abspath(slices_dir)
    
    if slicer_type == 'cf':
        # Use CheckerFrameworkSlicer
        cf_slicer_jar = os.path.join(cfwr_root, 'build/libs/cf-slicer-all.jar')
        if not os.path.exists(cf_slicer_jar):
            print(f"Error: CheckerFrameworkSlicer JAR not found at {cf_slicer_jar}")
            sys.exit(1)
        
        # Find all Java files in the project
        java_files = []
        for root, _, files in os.walk(project_root):
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))
        
        if not java_files:
            print("No Java files found in project")
            return
        
        # Run CheckerFrameworkSlicer
        cmd = ['java', '-cp', cf_slicer_jar, 'cfwr.CheckerFrameworkSlicer', 
               warnings_file, slices_dir] + java_files
        run(cmd, env=env)
    else:
        # Ensure Specimin has access to CF jars if not already provided
        try:
            if slicer_type == 'specimin':
                specimin_jarpath = env.get('SPECIMIN_JARPATH', '').strip()
                if not specimin_jarpath:
                    candidate_dirs = [
                        '/home/ubuntu/checker-framework-3.42.0/checker/dist',
                        '/home/ubuntu/checker-framework/checker/dist',
                        '/home/ubuntu/checker-framework/checker/build/libs',
                    ]
                    existing_dirs = [d for d in candidate_dirs if os.path.isdir(d)]
                    if existing_dirs:
                        env['SPECIMIN_JARPATH'] = os.pathsep.join(existing_dirs)
        except Exception:
            pass
        # Use runResolver task with proper argument handling
        args_str = f"{project_root} {warnings_file} {cfwr_root} {slicer_type}"
        run(['./gradlew', 'runResolver', f"-Pargs={args_str}"], env=env)


def run_cfg_generation(slices_dir, cfg_output_dir):
    print(f"=== CFG GENERATION DEBUG ===")
    print(f"Slices directory: {slices_dir}")
    print(f"CFG output directory: {cfg_output_dir}")
    print(f"Slices dir exists: {os.path.exists(slices_dir)}")
    
    if not os.path.exists(slices_dir):
        print(f"ERROR: Slices directory does not exist: {slices_dir}")
        return
    
    print(f"Contents of slices directory:")
    for item in os.listdir(slices_dir):
        item_path = os.path.join(slices_dir, item)
        print(f"  - {item} ({'dir' if os.path.isdir(item_path) else 'file'})")
    
    processed_files = 0
    for name in os.listdir(slices_dir):
        if not name.endswith('.java') and not os.path.isdir(os.path.join(slices_dir, name)):
            # Accept either raw .java files or per-slice directories created by Specimin
            continue
        path = os.path.join(slices_dir, name)
        print(f"Processing: {path}")
        
        if os.path.isdir(path):
            # Find any .java under this slice directory and generate CFGs per file
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith('.java'):
                        java_file = os.path.join(root, f)
                        base = os.path.splitext(os.path.basename(java_file))[0]
                        out_dir = os.path.join(cfg_output_dir, base)
                        if not os.path.exists(out_dir) or not any(n.endswith('.json') for n in os.listdir(out_dir)):
                            print(f"  Generating CFG for: {java_file}")
                            run([sys.executable, 'cfg.py', '--java_file', java_file, '--out_dir', cfg_output_dir])
                            processed_files += 1
                        else:
                            print(f"  CFG already exists for: {java_file}")
        else:
            java_file = path
            base = os.path.splitext(os.path.basename(java_file))[0]
            out_dir = os.path.join(cfg_output_dir, base)
            if not os.path.exists(out_dir) or not any(n.endswith('.json') for n in os.listdir(out_dir)):
                print(f"  Generating CFG for: {java_file}")
                run([sys.executable, 'cfg.py', '--java_file', java_file, '--out_dir', cfg_output_dir])
                processed_files += 1
            else:
                print(f"  CFG already exists for: {java_file}")
    
    print(f"Processed {processed_files} files for CFG generation")
    
    # Handle CheckerFrameworkSlicer output: copy existing CFG files
    print(f"Checking for existing CFG files from CheckerFrameworkSlicer...")
    cf_cfgs_found = 0
    for name in os.listdir(slices_dir):
        if name.endswith('_cfg.json'):
            # This is a CFG file generated by CheckerFrameworkSlicer
            cfg_file = os.path.join(slices_dir, name)
            base_name = name.replace('_cfg.json', '')
            target_dir = os.path.join(cfg_output_dir, base_name)
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy the CFG file to the expected location
            import shutil
            target_cfg = os.path.join(target_dir, 'cfg.json')
            if not os.path.exists(target_cfg):
                shutil.copy2(cfg_file, target_cfg)
                print(f"Copied CFG: {cfg_file} -> {target_cfg}")
                cf_cfgs_found += 1
            else:
                print(f"CFG already exists: {target_cfg}")
    
    print(f"Found and processed {cf_cfgs_found} existing CFG files from CheckerFrameworkSlicer")
    print(f"CFG generation completed!")


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
    parser.add_argument('--steps', default='all', choices=['all','slice','augment','cfg','train','predict','predict-original'], help='Which step to run')
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
    parser.add_argument('--augmented_dir', default=AUGMENTED_DIR_DEFAULT, help='Output directory for augmented slices')
    parser.add_argument('--augment_variants', type=int, default=10, help='Variants per original slice for augmentation')
    parser.add_argument('--slicer', default='cf', choices=['cf','wala','specimin'], help='Slicer to use (cf=CheckerFrameworkSlicer, wala, or specimin)')
    args = parser.parse_args()

    os.makedirs(args.slices_dir, exist_ok=True)
    os.makedirs(args.cfg_output_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.augmented_dir, exist_ok=True)

    if args.steps in ('slice','all'):
        if not args.project_root or not args.warnings_file:
            print('Error: --project_root and --warnings_file are required for slice step')
            sys.exit(2)
        run_slicing(args.project_root, args.warnings_file, args.cfwr_root, args.slices_dir, args.slicer)
        # Update slices_dir to point to the slicer-specific directory
        args.slices_dir = os.path.join(args.slices_dir, f"slices_{args.slicer}")

    if args.steps in ('cfg','all'):
        run_cfg_generation(args.slices_dir, args.cfg_output_dir)
        # Also generate CFGs for augmented slices if present
        augmented_dir = os.path.join(args.augmented_dir, f"slices_aug_{args.slicer}")
        if os.path.isdir(augmented_dir):
            run_cfg_generation(augmented_dir, args.cfg_output_dir)

    if args.steps in ('train','all'):
        run_train(args.model)

    if args.steps == 'predict':
        if not args.predict_java_file:
            print('Error: --predict_java_file is required when running predict step')
            sys.exit(2)
        run_predict(args.model, args.predict_java_file, args.models_dir, args.predict_out_dir)

    if args.steps == 'predict-original':
        run_predict_over_original(args.model, args.original_root, args.models_dir, args.predict_out_dir)

    if args.steps == 'augment':
        # Create slicer-specific augmented directory
        augmented_dir = os.path.join(args.augmented_dir, f"slices_aug_{args.slicer}")
        os.makedirs(augmented_dir, exist_ok=True)
        run([sys.executable, 'augment_slices.py', '--slices_dir', args.slices_dir, '--out_dir', augmented_dir, '--variants_per_file', str(args.augment_variants)])


if __name__ == '__main__':
    main()


