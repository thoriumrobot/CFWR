// Test case for Issue panacekcz#12:
// https://github.com/panacekcz/checker-framework/issues/12

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class RefineNeqLength {
    void refineNeqLength(int[] array, int i) {
        Contract.Requires(i >= 0 && i <= array.Length);
        // Refines i <= array.Length to i < array.Length
        if (i != array.Length) {
            refineNeqLengthMOne(array, i);
        }
        // No refinement
        if (i != array.Length - 1) {
            // :: error: (argument.type.incompatible)
            refineNeqLengthMOne(array, i);
        }
    }

    void refineNeqLengthMOne(int[] array, int i) {
        Contract.Requires(i >= 0 && i < array.Length);
        // Refines i < array.Length to i < array.Length - 1
        if (i != array.Length - 1) {
            refineNeqLengthMTwo(array, i);
            // :: error: (argument.type.incompatible)
            refineNeqLengthMThree(array, i);
        }
    }

    void refineNeqLengthMTwo(int[] array, int i) {
        Contract.Requires(i >= 0 && i < array.Length - 1);
        // Refines i < array.Length - 1 to i < array.Length - 2
        if (i != array.Length - 2) {
            refineNeqLengthMThree(array, i);
        }
        // No refinement
        if (i != array.Length - 1) {
            // :: error: (argument.type.incompatible)
            refineNeqLengthMThree(array, i);
        }
    }

    void refineNeqLengthMTwoNonLiteral(
            int[] array,
            int i,
            int c3,
            int c23) {
        Contract.Requires(i >= 0);
        Contract.Requires(i < array.Length - 1);
        Contract.Requires(c3 == 3);
        Contract.Requires(c23 == 2 || c23 == 3);

        // Refines i < array.Length - 1 to i < array.Length - 2
        if (i != array.Length - (5 - c3)) {
            refineNeqLengthMThree(array, i);
        }
        // No refinement
        if (i != array.Length - c23) {
            // :: error: (argument.type.incompatible)
            refineNeqLengthMThree(array, i);
        }
    }

    int refineNeqLengthMThree(
            int[] array, int i) {
        Contract.Requires(i >= 0);
        Contract.Requires(i + 2 < array.Length);
        Contract.Ensures(Contract.Result<int>() + 3 < array.Length);
        // Refines i < array.Length - 2 to i < array.Length - 3
        if (i != array.Length - 3) {
            return i;
        }
        // :: error: (return.type.incompatible)
        return i;
    }

    // The same test for a string.
    int refineNeqLengthMThree(
            String str, int i) {
        Contract.Requires(i >= 0);
        Contract.Requires(i + 2 < str.Length);
        Contract.Ensures(Contract.Result<int>() + 3 < str.Length);
        // Refines i < str.length() - 2 to i < str.length() - 3
        if (i != str.Length - 3) {
            return i;
        }
        // :: error: (return.type.incompatible)
        return i;
    }
}
