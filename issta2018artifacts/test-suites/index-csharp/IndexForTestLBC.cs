using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

/*@SuppressWarnings("upperbound")*/
public class IndexForTestLBC {
    int[] array = {0};

    void test1( int i) {
        Contract.Requires(i >= 0 && i < array.Length);
        int x = this.array[i];
    }

    void callTest1(int x) {
        test1(0);
        test1(1);
        test1(2);
        test1(array.Length);
        // :: error: (argument.type.incompatible)
        test1(array.Length - 1);
        if (array.Length > x) {
            // :: error: (argument.type.incompatible)
            test1(x);
        }

        if (array.Length == x) {
            test1(x);
        }
    }
}
