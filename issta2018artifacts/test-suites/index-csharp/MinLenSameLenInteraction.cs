using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class MinLenSameLenInteraction {
    void test(int [] a, int [] b) {
        Contract.Requires(a.Length == b.Length);
        if (a.Length == 1) {
            int x = b[0];
        }
    }
}
