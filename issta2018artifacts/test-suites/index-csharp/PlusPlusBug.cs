using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class PlusPlusBug {
    int[] array = {};

    void test(int x) {
        Contract.Requires(x < array.Length);
        // :: error: (compound.assignment.type.incompatible)
        x++;
        // :: error: (compound.assignment.type.incompatible)
        ++x;
        // :: error: (assignment.type.incompatible)
        x = x + 1;
    }
}
