// This test case is for issue 44: https://github.com/kelloggm/checker-framework/issues/44

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class ArrayCreationChecks {

    void test1(int x, int y) {
        Contract.Requires(x > 0);
        Contract.Requires(y > 0);
        int[] newArray = new int[x + y];
        int i = x;
        if(TestHelper.nondet()) Contract.Assert(i >= 0 && i < newArray.Length);
        int j = y;
        if(TestHelper.nondet()) Contract.Assert(j >= 0 && j < newArray.Length);
    }

    void test2(int x, int y) {
        Contract.Requires(x >= 0);
        Contract.Requires(y > 0);
        int[] newArray = new int[x + y];
        int i = x;
        if(TestHelper.nondet()) Contract.Assert(i >= 0 && i < newArray.Length);
        int j = y;
        if(TestHelper.nondet()) Contract.Assert(j >= 0 && j <= newArray.Length);
    }

    void test3(int x, int y) {
        Contract.Requires(x >= 0);
        Contract.Requires(y >= 0);
        int[] newArray = new int[x + y];
        int i = x;
        if(TestHelper.nondet()) Contract.Assert(i >= 0 && i <= newArray.Length);
        int j = y;
        if(TestHelper.nondet()) Contract.Assert(j >= 0 && j <= newArray.Length);
    }

    void test4(int x, int y) {
        Contract.Requires(x >= -1);
        Contract.Requires(y >= 0);
        // :: error: (array.Length.negative)
        int[] newArray = new int[x + y];
        int i = x;
        if(TestHelper.nondet()) Contract.Assert(i <= newArray.Length);
        // :: error: (assignment.type.incompatible)
        int j = y;
        if(TestHelper.nondet()) Contract.Assert(j >= 0 && j <= newArray.Length);
    }

    void test5(int x, int y) {
        Contract.Requires(x >= -1);
        Contract.Requires(y >= -1);
        // :: error: (array.Length.negative)
        int[] newArray = new int[x + y];
        // :: error: (assignment.type.incompatible)
        int i = x;
        if(TestHelper.nondet()) Contract.Assert(i >= 0 && i <= newArray.Length);
        // :: error: (assignment.type.incompatible)
        int j = y;
        if(TestHelper.nondet()) Contract.Assert(j >= 0 && j <= newArray.Length);
    }

    void test6(int x, int y) {
        // :: error: (array.Length.negative)
        int[] newArray = new int[x + y];
        // :: error: (assignment.type.incompatible)
        int i = x;
        if(TestHelper.nondet()) Contract.Assert(i >= 0 && i < newArray.Length);
        // :: error: (assignment.type.incompatible)
        int j = y;
        if(TestHelper.nondet()) Contract.Assert(j >= 0 && j <= newArray.Length);
    }
}
