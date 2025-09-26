using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class CastArray {
    void test(Object a) {
        int[] b = (int[]) a;
    }
}
