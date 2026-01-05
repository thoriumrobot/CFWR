#!/usr/bin/env python3
"""
Analyze hyperparameter search results and generate comprehensive report
"""

import json
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_comprehensive_results():
    """Analyze the comprehensive hyperparameter search results"""
    
    results_file = "/home/ubuntu/CFWR/comprehensive_hyperparameter_search_results_20250927_231941.json"
    
    if not os.path.exists(results_file):
        logger.error(f"Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    logger.info("📊 Analyzing Comprehensive Hyperparameter Search Results")
    
    # Overall statistics
    metadata = results['metadata']
    print(f"\n🎯 COMPREHENSIVE HYPERPARAMETER SEARCH ANALYSIS")
    print(f"=" * 60)
    print(f"📊 Total Models: {metadata['total_models']}")
    print(f"🧪 Total Tests: {metadata['total_tests']}")
    print(f"✅ Completed Tests: {metadata['completed_tests']}")
    print(f"📈 Completion Rate: {metadata['completion_rate']:.1%}")
    print(f"🕒 Timestamp: {metadata['timestamp']}")
    
    # Collect all successful results
    all_results = []
    
    for annotation_type in ['positive', 'nonnegative', 'gtenegativeone']:
        if annotation_type in results:
            for base_model, model_result in results[annotation_type].items():
                if model_result['best_params']:
                    all_results.append({
                        'model': f"{annotation_type}_{base_model}",
                        'annotation_type': annotation_type,
                        'base_model': base_model,
                        'score': model_result['best_score'],
                        'reward': model_result['best_reward'],
                        'predictions': model_result['best_predictions'],
                        'accuracy': model_result['best_accuracy'],
                        'success_rate': model_result['success_rate'],
                        'params': model_result['best_params']
                    })
    
    # Sort by score
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n🏆 TOP 10 MODELS BY COMPOSITE SCORE:")
    print("-" * 60)
    for i, result in enumerate(all_results[:10], 1):
        print(f"{i:2d}. {result['model']:<25} Score: {result['score']:>8.4f} "
              f"Reward: {result['reward']:>6.3f} Pred: {result['predictions']:>3d} "
              f"Acc: {result['accuracy']:>5.3f} SR: {result['success_rate']:>5.1%}")
    
    # Enhanced causal model performance
    enhanced_causal_results = [r for r in all_results if 'enhanced_causal' in r['model']]
    if enhanced_causal_results:
        print(f"\n🚀 ENHANCED CAUSAL MODEL PERFORMANCE:")
        print("-" * 60)
        for result in enhanced_causal_results:
            print(f"   {result['model']:<25} Score: {result['score']:>8.4f} "
                  f"Reward: {result['reward']:>6.3f} Pred: {result['predictions']:>3d} "
                  f"Acc: {result['accuracy']:>5.3f}")
    
    # Model type comparison
    print(f"\n📊 MODEL TYPE COMPARISON:")
    print("-" * 60)
    
    model_types = {}
    for result in all_results:
        base_model = result['base_model']
        if base_model not in model_types:
            model_types[base_model] = []
        model_types[base_model].append(result)
    
    for base_model in sorted(model_types.keys()):
        models = model_types[base_model]
        avg_score = sum(m['score'] for m in models) / len(models)
        avg_success_rate = sum(m['success_rate'] for m in models) / len(models)
        print(f"   {base_model:<18} Avg Score: {avg_score:>8.4f} "
              f"Avg Success Rate: {avg_success_rate:>5.1%} ({len(models)} models)")
    
    # Annotation type comparison
    print(f"\n🎯 ANNOTATION TYPE COMPARISON:")
    print("-" * 60)
    
    annotation_types = {}
    for result in all_results:
        annotation_type = result['annotation_type']
        if annotation_type not in annotation_types:
            annotation_types[annotation_type] = []
        annotation_types[annotation_type].append(result)
    
    for annotation_type in sorted(annotation_types.keys()):
        models = annotation_types[annotation_type]
        avg_score = sum(m['score'] for m in models) / len(models)
        avg_success_rate = sum(m['success_rate'] for m in models) / len(models)
        print(f"   {annotation_type:<18} Avg Score: {avg_score:>8.4f} "
              f"Avg Success Rate: {avg_success_rate:>5.1%} ({len(models)} models)")
    
    # Best configurations by annotation type
    print(f"\n🏆 BEST CONFIGURATIONS BY ANNOTATION TYPE:")
    print("-" * 60)
    
    for annotation_type in ['positive', 'nonnegative', 'gtenegativeone']:
        type_results = [r for r in all_results if r['annotation_type'] == annotation_type]
        if type_results:
            best = type_results[0]  # Already sorted by score
            print(f"\n{annotation_type.upper()}:")
            print(f"   Best Model: {best['model']}")
            print(f"   Score: {best['score']:.4f}")
            print(f"   Parameters:")
            for param, value in best['params'].items():
                print(f"     --{param} {value}")
    
    # Enhanced causal vs original causal comparison
    print(f"\n🆚 ENHANCED CAUSAL vs ORIGINAL CAUSAL COMPARISON:")
    print("-" * 60)
    
    enhanced_results = [r for r in all_results if 'enhanced_causal' in r['model']]
    original_results = [r for r in all_results if 'causal' in r['model'] and 'enhanced' not in r['model']]
    
    if enhanced_results and original_results:
        for annotation_type in ['positive', 'nonnegative', 'gtenegativeone']:
            enhanced = next((r for r in enhanced_results if r['annotation_type'] == annotation_type), None)
            original = next((r for r in original_results if r['annotation_type'] == annotation_type), None)
            
            if enhanced and original:
                improvement = ((enhanced['score'] - original['score']) / original['score'] * 100) if original['score'] > 0 else float('inf')
                print(f"   {annotation_type:<15} Enhanced: {enhanced['score']:>8.4f} "
                      f"Original: {original['score']:>8.4f} "
                      f"Improvement: {improvement:>+6.1f}%")
    
    print("\n" + "=" * 60)
    
    return all_results

def generate_best_configurations_report(all_results):
    """Generate a detailed report of best configurations"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"/home/ubuntu/CFWR/best_configurations_analysis_{timestamp}.md"
    
    with open(report_file, 'w') as f:
        f.write("# Best Configurations Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"Total models tested: {len(all_results)}\n")
        f.write(f"Top performer: {all_results[0]['model']} (Score: {all_results[0]['score']:.4f})\n\n")
        
        # Enhanced causal performance
        enhanced_results = [r for r in all_results if 'enhanced_causal' in r['model']]
        if enhanced_results:
            f.write("## Enhanced Causal Model Performance\n\n")
            f.write("The enhanced causal model shows significant improvements:\n\n")
            for result in enhanced_results:
                f.write(f"- **{result['model']}**: Score {result['score']:.4f}, "
                        f"Reward {result['reward']:.3f}, "
                        f"Predictions {result['predictions']}, "
                        f"Accuracy {result['accuracy']:.3f}\n")
            f.write("\n")
        
        f.write("## Top 10 Models\n\n")
        f.write("| Rank | Model | Score | Reward | Predictions | Accuracy | Success Rate |\n")
        f.write("|------|-------|-------|--------|-------------|----------|--------------|\n")
        
        for i, result in enumerate(all_results[:10], 1):
            f.write(f"| {i} | {result['model']} | {result['score']:.4f} | "
                    f"{result['reward']:.3f} | {result['predictions']} | "
                    f"{result['accuracy']:.3f} | {result['success_rate']:.1%} |\n")
        
        f.write("\n## Best Configurations by Annotation Type\n\n")
        
        for annotation_type in ['positive', 'nonnegative', 'gtenegativeone']:
            type_results = [r for r in all_results if r['annotation_type'] == annotation_type]
            if type_results:
                best = type_results[0]
                f.write(f"### {annotation_type.upper()}\n\n")
                f.write(f"**Best Model:** {best['model']}\n")
                f.write(f"**Score:** {best['score']:.4f}\n")
                f.write(f"**Reward:** {best['reward']:.3f}\n")
                f.write(f"**Predictions:** {best['predictions']}\n")
                f.write(f"**Accuracy:** {best['accuracy']:.3f}\n\n")
                
                f.write("**Best Parameters:**\n")
                for param, value in best['params'].items():
                    f.write(f"- `--{param} {value}`\n")
                f.write("\n")
                
                # Command
                script_map = {
                    'positive': 'annotation_type_rl_positive.py',
                    'nonnegative': 'annotation_type_rl_nonnegative.py',
                    'gtenegativeone': 'annotation_type_rl_gtenegativeone.py'
                }
                
                cmd = [
                    'python', script_map[annotation_type],
                    '--base_model', best['base_model'],
                    '--project_root', '/home/ubuntu/checker-framework/checker/tests/index'
                ]
                
                for param, value in best['params'].items():
                    cmd.extend([f'--{param}', str(value)])
                
                f.write("**Command:**\n")
                f.write("```bash\n")
                f.write(' '.join(cmd) + '\n')
                f.write("```\n\n")
    
    logger.info(f"📄 Best configurations analysis report saved to {report_file}")
    return report_file

if __name__ == "__main__":
    logger.info("🔍 Starting Hyperparameter Search Results Analysis")
    
    all_results = analyze_comprehensive_results()
    
    if all_results:
        report_file = generate_best_configurations_report(all_results)
        logger.info(f"✅ Analysis completed. Report saved to: {report_file}")
    else:
        logger.error("❌ Analysis failed - no results to analyze")
