using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class LubIndex {

    //CHANGE: added bool parameter to avoid unreachable code
    public static void MinLen(int [] arg, int [] arg2, bool y) {
        Contract.Requires(arg.Length >= 10);
        Contract.Requires(arg2.Length >= 4);

        int[] arr;
        if (y) {
            arr = arg;
        } else {
            arr = arg2;
        }
        // :: error: (assignment.type.incompatible)
        int[] res = arr;
        if(TestHelper.nondet()) Contract.Assert(res.Length >= 10);
        int[] res2 = arr;
        if(TestHelper.nondet()) Contract.Assert(res2.Length >= 4);
    #if BOTTOM
        // :: error: (assignment.type.incompatible)
        int @BottomVal [] res3 = arr;
    #endif
    }
    #if BOTTOM
    public static void Bottom(int @BottomVal [] arg, int @MinLen(4) [] arg2) {
        int[] arr;
        if (true) {
            arr = arg;
        } else {
            arr = arg2;
        }
        // :: error: (assignment.type.incompatible)
        int @MinLen(10) [] res = arr;
        int @MinLen(4) [] res2 = arr;
        // :: error: (assignment.type.incompatible)
        int @BottomVal [] res3 = arr;
    }

    public static void BothBottom(int @BottomVal [] arg, int @BottomVal [] arg2) {
        int[] arr;
        if (true) {
            arr = arg;
        } else {
            arr = arg2;
        }
        int @MinLen(10) [] res = arr;
        int @BottomVal [] res2 = arr;
    }
    #endif
}
