using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class OneOrTwo {
    int getOneOrTwo() {
        Contract.Ensures(Contract.Result<int>() >= 1 && Contract.Result<int>() <= 2);
        return 1;
    }
#if BOTTOM
    void test(@BottomVal int x) {
        int[] a = new int[Integer.valueOf(getOneOrTwo())];
        // :: error: (array.Length.negative)
        int[] b = new int[Integer.valueOf(x)];
    }
#endif
#if POLY
    @PolyValue int poly(@PolyValue int y) {
        return y;
    }
#endif
}
