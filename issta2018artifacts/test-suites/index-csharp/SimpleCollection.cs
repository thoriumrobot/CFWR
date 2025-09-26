using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class SimpleCollection {
    protected int[] values;

    int size() {
        Contract.Ensures(Contract.Result<int>() >= 0 && Contract.Result<int>() <= values.Length);
        return values.Length;
    }

    void interact_with_other(SimpleCollection other) {
        int[] othervalues = other.values;
        int [] x = othervalues;
        if(TestHelper.nondet()) Contract.Assert(x.Length == other.values.Length);

        for (int i = 0; i < other.size(); i++) {
            int k = othervalues[i];
        }
        for (int j = 0; j < other.size(); j++) {
            int k = other.values[j];
        }
    }
}
