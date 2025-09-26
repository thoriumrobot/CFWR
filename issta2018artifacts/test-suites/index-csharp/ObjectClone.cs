// Test case for issue 146: https://github.com/kelloggm/checker-framework/issues/146

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class ObjectClone {

    void test(int[] a, int [] b) {
        Contract.Requires(b.Length == a.Length);
        int[] c = (int[])b.Clone();
        if(TestHelper.nondet()) Contract.Assert(c.Length == a.Length);
        int[] d = (int[])b.Clone();
        if(TestHelper.nondet()) Contract.Assert(d.Length == a.Length && d.Length == d.Length);
        int[] e = b;
        if(TestHelper.nondet()) Contract.Assert(e.Length == a.Length && e.Length == e.Length);
        int[] f = b;
        if(TestHelper.nondet()) Contract.Assert(f.Length == f.Length);
    }

    public static void main(String[] args) {
        String[] args2 = args;
        if(TestHelper.nondet()) Contract.Assert(args2.Length == args.Length);
        String[] args_sorted = (string[])args.Clone();
        if(TestHelper.nondet()) Contract.Assert(args_sorted.Length == args.Length && args_sorted.Length == args_sorted.Length);
        Array.Sort(args_sorted);
        String[] args_sorted2 = (string[])args_sorted.Clone();
        if(TestHelper.nondet()) Contract.Assert(args_sorted2.Length == args.Length && args_sorted2.Length == args_sorted.Length);
        if (args_sorted.Length == 1) {
            int i = 0;
            if(TestHelper.nondet()) Contract.Assert(i >= 0 && i < args_sorted.Length);
            int j = 0;
            if(TestHelper.nondet()) Contract.Assert(j >= 0 && j < args.Length);
            String[] k = args;
            if(TestHelper.nondet()) Contract.Assert(k.Length == args.Length && k.Length == args_sorted.Length);
            Console.WriteLine(args[0]);
        }
    }
}
