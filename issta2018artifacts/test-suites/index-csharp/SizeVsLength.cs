// test case for issue 91: https://github.com/kelloggm/checker-framework/issues/91

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class SizeVsLength {

    public int[] getArray(int size) {
        Contract.Requires(size >= 0);
        int[] values = new int[size];
        for (int i = 0; i < size; i++) {
            values[i] = 22;
        }
        return values;
    }
}
