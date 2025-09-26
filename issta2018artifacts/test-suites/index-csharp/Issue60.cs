// Testcase for Issue 60
// https://github.com/kelloggm/checker-framework/issues/60

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class Issue60 {

    public static int[] fn_compose(int[] a, int[] b) {
        Contract.ForAll(0, a.Length, i => a[i] >= 0 && a[i] < b.Length);
        int[] result = new int[a.Length];
        for (int i = 0; i < a.Length; i++) {
            int inner = a[i];
            if (inner == -1) {
                result[i] = -1;
            } else {
                result[i] = b[inner];
            }
        }
        return result;
    }
}
