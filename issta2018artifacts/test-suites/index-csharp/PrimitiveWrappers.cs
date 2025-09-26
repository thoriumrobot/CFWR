// Test for issue 65: https://github.com/kelloggm/checker-framework/issues/65

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;


// This test ensures that the checker functions on primitive wrappers in
// addition to literal primitives. Primarily it focuses on Integer/int.

class PrimitiveWrappers {

    void int_Integer_access_equivalent(int? i, int j, int[] a) {
        Contract.Requires(i.Value >= 0 && i.Value < a.Length);
        Contract.Requires(j >= 0 && j < a.Length);
        a[i.Value] = a[j];
    }

    void array_creation(int? i, int j) {
        Contract.Requires(i.Value >= 0);
        Contract.Requires(j >= 0);
        int[] a = new int[j];
        int[] b = new int[i.Value];
    }
}
