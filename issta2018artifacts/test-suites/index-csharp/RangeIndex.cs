using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class RangeIndex {
    void foo(int x, int [] a) {
        Contract.Requires(x >= 0 && x <= 11);
        Contract.Requires(a.Length >= 10);
        // :: error: (array.access.unsafe.high.range)
        int y = a[x];
    }
}
