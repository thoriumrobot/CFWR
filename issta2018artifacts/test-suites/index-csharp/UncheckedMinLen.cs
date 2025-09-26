using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

// test case for kelloggm#183: https://github.com/kelloggm/checker-framework/issues/183

public class UncheckedMinLen {
    void addToNonNegative(int l, Object v) {
        Contract.Requires(l >= 0);
        // :: error: (assignment.type.incompatible)
        Object[] o = new Object[l + 1];
        if(TestHelper.nondet()) Contract.Assert(o.Length >= 100);
        o[99] = v;
    }

    void addToPositive(int l, Object v) {
        Contract.Requires(l > 0);
        // :: error: (assignment.type.incompatible)
        Object[] o = new Object[l + 1];
        if(TestHelper.nondet()) Contract.Assert(o.Length >= 100);
        o[99] = v;
    }

    void addToUnboundedIntRange(int l, Object v) {
        Contract.Requires(l >= 0);
        // :: error: (assignment.type.incompatible)
        Object[] o = new Object[l + 1];
        if(TestHelper.nondet()) Contract.Assert(o.Length >= 100);
        o[99] = v;
    }

    // Similar code that correctly gives warnings
    void addToPositiveOK(int l, Object v) {
        Contract.Requires(l >= 0);
        Object[] o = new Object[l + 1];
        // :: error: (array.access.unsafe.high.constant)
        o[99] = v;
    }

    void addToBoundedIntRangeOK(int l, Object v) {
        Contract.Requires(l >= 0 && l <= 1);
        // :: error: (assignment.type.incompatible)
        Object[] o = new Object[l + 1];
        if(TestHelper.nondet()) Contract.Assert(o.Length >= 100);
        o[99] = v;
    }

    void subtractFromPositiveOK(int l, Object v) {
        Contract.Requires(l > 0);
        // :: error: (assignment.type.incompatible)
        Object[] o = new Object[l - 1];
        if(TestHelper.nondet()) Contract.Assert(o.Length >= 100);
        o[99] = v;
    }
}
