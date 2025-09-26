# CFWR Data Augmentation Methods

## Overview

The CFWR (Checker Framework Warning Resolver) project implements data augmentation techniques to generate diverse training data for machine learning models. The augmentation system creates syntactically correct Java code variations while preserving the original Checker Framework annotations and logic that are essential for training.

## Core Philosophy

The augmentation approach is designed to:
- **Preserve Semantic Meaning**: Original Checker Framework logic remains intact
- **Generate True Variety**: Each augmentation produces genuinely unique code
- **Maintain Syntactic Correctness**: All generated code compiles successfully
- **Enhance ML Training**: Provide diverse patterns for model learning

## Augmentation Methods

### 1. Random Method Generation (`generate_random_method()`)

**Purpose**: Creates new Java methods with random signatures and implementations.

**Components**:
- **Access Modifiers**: Randomly selects from `private`, `public`, `protected`, or no modifier
- **Static Modifiers**: Randomly includes or omits `static` keyword
- **Return Types**: Chooses from Java primitive types (`int`, `long`, `double`, `float`, `boolean`, `char`, `byte`, `short`) and reference types (`String`, `Object`, `Integer`, `Long`, `Double`, `Float`, `Boolean`, `Character`)
- **Method Names**: Generates unique names using pattern `__cfwr_{method_type}{random_number}` where method types include `helper`, `util`, `temp`, `aux`, `proc`, `func`, `calc`, `compute`, `process`, `handle`
- **Parameters**: Randomly generates 0-3 parameters with random types and names following pattern `__cfwr_p{index}`
- **Method Body**: Creates 1-4 random statements plus appropriate return statement

**Example Output**:
```java
public static String __cfwr_helper123(int __cfwr_p0, Object __cfwr_p1) {
    double __cfwr_val42 = 45.67;
    if (true && false) {
        char __cfwr_data15 = 'A';
    }
    return "hello99";
}
```

### 2. Random Statement Generation (`generate_random_statement()`)

**Purpose**: Creates individual Java statements that can be inserted into existing methods.

**Statement Types**:

#### A. Variable Assignment Statements
- **Pattern**: `{type} {variable_name} = {expression};`
- **Variable Names**: Generated using pattern `__cfwr_{name_type}{random_number}` where name types include `val`, `data`, `item`, `obj`, `result`, `temp`, `var`, `elem`, `node`, `entry`
- **Types**: Randomly selected from primitive and reference types
- **Expressions**: Generated using `generate_random_expression()`

**Example**: `int __cfwr_val42 = 123;`

#### B. Control Flow Statements

**If Statements**:
- **Pattern**: `if ({condition}) { {statement} }`
- **Conditions**: Generated using boolean expressions with logical operators (`&&`, `||`)
- **Body**: Contains one random statement

**Example**:
```java
if ((true && false) || true) {
    String __cfwr_data71 = "world80";
}
```

**For Loops**:
- **Pattern**: `for (int {var_name} = 0; {var_name} < {limit}; {var_name}++) { {statement} }`
- **Variable Names**: Generated using pattern `__cfwr_i{random_number}`
- **Limits**: Random integers from 1-10
- **Body**: Contains one random statement

**Example**:
```java
for (int __cfwr_i48 = 0; __cfwr_i48 < 3; __cfwr_i48++) {
    String __cfwr_data71 = "world80";
}
```

**While Loops**:
- **Pattern**: `while ({condition}) { {statement} break; }`
- **Conditions**: Random boolean expressions
- **Safety**: Always includes `break;` to prevent infinite loops

**Example**:
```java
while (false) {
    char __cfwr_result41 = 'v';
    break; // Prevent infinite loops
}
```

#### C. Exception Handling Statements

**Try-Catch Blocks**:
- **Pattern**: `try { {statement} } catch (Exception {exception_name}) { // ignore }`
- **Exception Names**: Generated using pattern `__cfwr_e{random_number}`
- **Body**: Contains one random statement
- **Catch Block**: Simple comment to ignore exceptions

**Example**:
```java
try {
    char __cfwr_result41 = 'v';
} catch (Exception __cfwr_e60) {
    // ignore
}
```

#### D. Return Statements
- **Pattern**: `return {expression};`
- **Expression**: Generated using `generate_random_expression()` with appropriate return type

**Example**: `return "hello99";`

### 3. Random Expression Generation (`generate_random_expression()`)

**Purpose**: Creates complex expressions with proper type handling and recursion control.

**Features**:
- **Recursion Control**: Maximum depth of 2 to prevent infinite recursion
- **Complexity Probability**: 30% chance to create complex expressions when depth < 2
- **Type Safety**: Ensures expressions match expected types
- **Operator Variety**: Uses arithmetic (`+`, `-`, `*`, `/`, `%`), bitwise (`&`, `|`, `^`, `<<`, `>>`), and logical operators (`&&`, `||`)

**Example**: `(123 + (45.67 * 89))`

### 4. Random Literal Generation (`generate_random_literal()`)

**Purpose**: Generates appropriate literal values for different Java types.

**Type-Specific Literals**:
- **int**: Random integers from -1000 to 1000
- **long**: Random integers with 'L' suffix
- **double**: Random decimal values with 2 decimal places
- **float**: Random decimal values with 'f' suffix
- **boolean**: `true` or `false`
- **char**: Random alphanumeric characters in single quotes
- **String**: Random words with numbers in double quotes
- **Reference Types**: `null`

**Examples**:
- `int`: `123`, `-456`
- `long`: `789L`, `-101L`
- `double`: `45.67`, `-89.12`
- `float`: `23.45f`, `-67.89f`
- `boolean`: `true`, `false`
- `char`: `'A'`, `'z'`, `'5'`
- `String`: `"hello42"`, `"world99"`

### 5. Insertion Strategies

#### A. Method Insertion (`insert_random_methods()`)
- **Location**: Before the last closing brace of the class
- **Count**: Random number of methods (1-3 by default)
- **Integration**: Seamlessly integrates with existing class structure

#### B. Statement Insertion (`insert_random_statements()`)
- **Location**: After method opening braces (`{`)
- **Target**: First few methods in the class
- **Count**: Random number of statements (1-2 by default)
- **Integration**: Adds statements at the beginning of method bodies

## Configuration and Control

### Command-Line Parameters

```bash
python augment_slices.py --slices_dir slices --out_dir slices_aug \
    --variants_per_file 10 \
    --seed 42 \
    --max_methods 3 \
    --max_statements 2
```

**Parameters**:
- `--slices_dir`: Directory containing original slice files
- `--out_dir`: Output directory for augmented files
- `--variants_per_file`: Number of variants to generate per original file
- `--seed`: Random seed for reproducible results
- `--max_methods`: Maximum number of random methods to add per file
- `--max_statements`: Maximum number of random statements to add per file

### Reproducibility

The system uses seeded random number generation to ensure:
- **Consistent Results**: Same seed produces identical augmentations
- **Controlled Variation**: Different seeds produce different but predictable variations
- **Debugging Support**: Easy to reproduce specific augmentation patterns

## Quality Assurance

### Syntactic Correctness
- All generated code follows Java syntax rules
- Proper type matching in expressions and assignments
- Correct method signatures and return statements
- Valid control flow structures

### Safety Measures
- **Infinite Loop Prevention**: While loops always include `break;` statements
- **Exception Safety**: Try-catch blocks handle exceptions gracefully
- **Type Safety**: Expressions are type-checked before generation
- **Recursion Limits**: Expression generation has depth limits

### Naming Conventions
- **Unique Identifiers**: All generated names use `__cfwr_` prefix to avoid conflicts
- **Descriptive Patterns**: Method and variable names follow logical patterns
- **Collision Avoidance**: Random numbers ensure uniqueness

## Benefits for Machine Learning

### 1. True Diversity
- Each augmentation is genuinely unique
- No template-based repetition
- Unlimited variation possibilities

### 2. Realistic Patterns
- Mimics real Java code structures
- Includes common programming patterns
- Covers various control flow scenarios

### 3. Robust Training
- Models learn to handle diverse code patterns
- Reduces overfitting to specific templates
- Improves generalization capabilities

### 4. Context Preservation
- Original Checker Framework logic remains intact
- Annotations and warnings are preserved
- Semantic meaning is maintained

## Example Augmentation Flow

**Input Slice**:
```java
import org.checkerframework.checker.index.qual.*;

public class ArrayTest {
    void test(@LTLengthOf("#1") int index) {
        int[] arr = new int[10];
        arr[index] = 42;
    }
}
```

**Augmented Output**:
```java
/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class ArrayTest {
    void test(@LTLengthOf("#1") int index) {
        String __cfwr_data42 = "hello99";
        
        int[] arr = new int[10];
        arr[index] = 42;
    }
    
    private static boolean __cfwr_helper123(int __cfwr_p0, String __cfwr_p1) {
        double __cfwr_val45 = 67.89;
        if (true && false) {
            char __cfwr_item12 = 'A';
        }
        return false;
    }
    
    public Object __cfwr_util456() {
        for (int __cfwr_i78 = 0; __cfwr_i78 < 5; __cfwr_i78++) {
            long __cfwr_temp34 = 123L;
        }
        return null;
    }
}
```

## Conclusion

The CFWR augmentation system provides an approach to generating diverse training data while maintaining the integrity of Checker Framework-specific code. By combining random generation with syntactic correctness and semantic preservation, it creates an environment for training machine learning models that can handle the complexity and variety of real-world Java code with type annotations.
