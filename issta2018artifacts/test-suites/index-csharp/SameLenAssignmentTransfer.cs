using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;


class SameLenAssignmentTransfer {
    void transfer5(int [] a, int[] b) {
        Contract.Requires(a.Length == b.Length);
        int[] c = a;
        for (int i = 0; i < c.Length; i++) { // i's type is @LTL("c")
            b[i] = 1;
        }
    }
}
