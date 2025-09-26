// Test case for issue #68:
// https://github.com/kelloggm/checker-framework/issues/68

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class TestAgainstLength {

    protected int[] values;
    /** The number of active elements (equivalently, the first unused index). */
    int num_values;

    public void add(int elt) {
        if (num_values == values.Length) {
            return;
        }
        values[num_values] = elt;
        num_values++;
    }

    [ContractInvariantMethod]
    private void Invariant()
    {
        Contract.Invariant(num_values >= 0 && num_values <= values.Length);
    }
}
