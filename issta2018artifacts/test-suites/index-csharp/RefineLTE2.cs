// Test case for issue #62:
// https://github.com/kelloggm/checker-framework/issues/62

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

/*@SuppressWarnings("lowerbound")*/
public class RefineLTE2 {

    protected int [] values;

    int num_values;

    [ContractInvariantMethod]
    private void Invariant()
    {
        Contract.Invariant(num_values <= values.Length);
        Contract.Invariant(values.Length >= 1);
    }



    public void add(int elt) {
        if (num_values == values.Length) {
            values = null;
            // :: error: (compound.assignment.type.incompatible)
            num_values++;
            return;
        }
        values[num_values] = elt;
        num_values++;
    }
}
