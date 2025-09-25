import os
import re
import argparse
import random
from pathlib import Path

HEADER_COMMENT = """
/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
""".strip()

DUMMY_METHOD_TEMPLATES = [
    """
    private static int __cfwr_helper_{idx}(int x) {
        int y = x;
        for (int i = 0; i < 3; i++) { y += i; }
        try { y += 0; } catch (Exception e) { y -= 0; }
        return y - x;
    }
    """,
    """
    private static String __cfwr_str_{idx}(String s) {
        if (s == null) { return ""; }
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) { if (c == '\\0') { break; } }
        return sb.toString();
    }
    """,
]

IF_FALSE_BLOCK = """
if (false) {
    int __cfwr_a = 0;
    int __cfwr_b = 1;
    __cfwr_a += __cfwr_b;
}
""".strip()


def find_class_insertion_point(src: str) -> int:
    # Heuristic: insert before the last closing brace of the file
    last = src.rfind('}')
    return last if last != -1 else len(src)


def insert_header_comment(src: str) -> str:
    if src.lstrip().startswith("/* CFWR augmentation"):
        return src
    return HEADER_COMMENT + "\n" + src


def insert_dummy_methods(src: str, count: int) -> str:
    insert_at = find_class_insertion_point(src)
    methods = []
    for i in range(count):
        tmpl = random.choice(DUMMY_METHOD_TEMPLATES)
        # Replace {idx} placeholder with random number
        idx_value = random.randint(1000, 9999)
        method_text = tmpl.replace('{idx}', str(idx_value))
        methods.append(method_text)
    addition = "\n".join(methods) + "\n"
    return src[:insert_at] + addition + src[insert_at:]


def insert_if_false_blocks(src: str, blocks: int) -> str:
    # Insert after method opening braces for first few methods
    out = src
    pattern = re.compile(r"(\)\s*\{)")
    matches = list(pattern.finditer(out))
    # Insert up to 'blocks' occurrences
    for m in matches[:blocks]:
        idx = m.end()
        out = out[:idx] + "\n" + IF_FALSE_BLOCK + "\n" + out[idx:]
    return out


def augment_file(java_path: str, variant_idx: int) -> str:
    with open(java_path, 'r') as f:
        src = f.read()
    src = insert_header_comment(src)
    src = insert_dummy_methods(src, count=random.randint(1, 3))
    src = insert_if_false_blocks(src, blocks=random.randint(1, 2))
    return src


def write_variant(original_path: str, out_dir: str, variant_idx: int):
    rel = os.path.basename(original_path)
    base = os.path.splitext(rel)[0]
    variant_dir = os.path.join(out_dir, f"{base}__aug{variant_idx}")
    os.makedirs(variant_dir, exist_ok=True)
    out_path = os.path.join(variant_dir, rel)
    augmented = augment_file(original_path, variant_idx)
    with open(out_path, 'w') as f:
        f.write(augmented)
    return out_path


def iter_java_files(root_dir: str):
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.java'):
                yield os.path.join(root, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slices_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--variants_per_file', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(42)

    produced = []
    for java_file in iter_java_files(args.slices_dir):
        for k in range(args.variants_per_file):
            out_path = write_variant(java_file, args.out_dir, k)
            produced.append(out_path)

    print(f"Augmented {len(produced)} files into {args.out_dir}")


if __name__ == '__main__':
    main()


