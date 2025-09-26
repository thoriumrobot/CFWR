// Test case for Issue 167:
// https://github.com/kelloggm/checker-framework/issues/167

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class Index167 {
    static void fn1(int[] arr, int i) {
        Contract.Requires(i >= 0 && i < arr.Length);
        if (i >= 33) {
            // :: error: (argument.type.incompatible)
            fn2(arr, i);
        }
        if (i > 33) {
            // :: error: (argument.type.incompatible)
            fn2(arr, i);
        }
        if (i != 33) {
            // :: error: (argument.type.incompatible)
            fn2(arr, i);
        }
    }

    static void fn2(int[] arr, int i) {
        Contract.Requires(i >= 0 && i<arr.Length - 1);
        int c = arr[i + 1];
    }
}
