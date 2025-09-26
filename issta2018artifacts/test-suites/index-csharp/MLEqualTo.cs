using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class MLEqualTo {

    public static void equalToMinLen(int [] m, int [] r) {
        Contract.Requires(m.Length >= 2);
        Contract.Requires(r.Length >= 0);
        if (r == m) {
            int[] j = r;
            if(TestHelper.nondet()) Contract.Assert(j.Length >= 2);
        }
    }
}
