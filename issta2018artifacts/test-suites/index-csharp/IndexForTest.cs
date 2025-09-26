using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class IndexForTest {
    int [] array = {0};

    [ContractInvariantMethod]
    private void Invariant()
    {
        Contract.Invariant(array.Length >= 1);
    }


    void test1(int i) {
        Contract.Requires(i >= 0 && i < array.Length);
        int x = array[i];
    }

    void callTest1(int x) {
        test1(0);
        // ::  error: (argument.type.incompatible)
        test1(1);
        // ::  error: (argument.type.incompatible)
        test1(2);
        // ::  error: (argument.type.incompatible)
        test1(array.Length);

        if (array.Length > 0) {
            test1(array.Length - 1);
        }

        test1(array.Length - 1);

        // ::  error: (argument.type.incompatible)
        test1(this.array.Length);

        if (array.Length > 0) {
            test1(this.array.Length - 1);
        }

        test1(this.array.Length - 1);

        if (this.array.Length > x && x >= 0) {
            test1(x);
        }

        if (array.Length == x) {
            // ::  error: (argument.type.incompatible)
            test1(x);
        }
    }

    void test2(int i) {
        Contract.Requires(i >= 0 && i< this.array.Length);
        int x = array[i];
    }

    void callTest2(int x) {
        test2(0);
        // ::  error: (argument.type.incompatible)
        test2(1);
        // ::  error: (argument.type.incompatible)
        test2(2);
        // ::  error: (argument.type.incompatible)
        test2(array.Length);

        if (array.Length > 0) {
            test2(array.Length - 1);
        }

        test2(array.Length - 1);

        // ::  error: (argument.type.incompatible)
        test2(this.array.Length);

        if (array.Length > 0) {
            test2(this.array.Length - 1);
        }

        test2(this.array.Length - 1);

        if (array.Length == x && x >= 0) {
            // ::  error: (argument.type.incompatible)
            test2(x);
        }
    }
}
