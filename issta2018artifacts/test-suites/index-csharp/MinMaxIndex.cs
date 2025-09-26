// Tests handling Math.Min and Math.Max methods.
// The upper bound of Math.Max is issue panacekcz#20:
// https://github.com/panacekcz/checker-framework/issues/20

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class MinMaxIndex {
    // Both min and max preserve IndexFor
    void indexFor(char[] array, int i1, int i2) {
        Contract.Requires(i1 >= 0 && i1 < array.Length);
        Contract.Requires(i2 >= 0 && i2 < array.Length);
        char c = array[Math.Max(i1, i2)];
        char d = array[Math.Min(i1, i2)];
    }
    // Both min and max preserve IndexOrHigh
    void indexOrHigh(String str, int i1, int i2) {
        Contract.Requires(i1 >= 0 && i1 <= str.Length);
        Contract.Requires(i2 >= 0 && i2 <= str.Length);
        str.Substring(Math.Max(i1, i2));
        str.Substring(Math.Min(i1, i2));
    }
    // Combining IndexFor and IndexOrHigh
    void indexForOrHigh(String str, int i1, int i2) {
        Contract.Requires(i1 >= 0 && i1 < str.Length);
        Contract.Requires(i2 >= 0 && i2 <= str.Length);
        str.Substring(Math.Max(i1, i2));
        str.Substring(Math.Min(i1, i2));
        // :: error: (argument.type.incompatible)
        var _1 = str[(Math.Max(i1, i2))];
        var _2 = str[(Math.Min(i1, i2))];
    }
    // max does not work with different sequences, min does
    void twoSequences(String str1, String str2, int i1, int i2) {
        Contract.Requires(i1 >= 0 && i1 < str1.Length);
        Contract.Requires(i2 >= 0 && i2 < str2.Length);
        // :: error: (argument.type.incompatible)
        var _1 = str1[(Math.Max(i1, i2))];
        var _2 = str1[(Math.Min(i1, i2))];
    }
}
