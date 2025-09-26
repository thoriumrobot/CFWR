using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class RandomTestLBC {
    void test() {
        Random rand = new Random();
        int[] a = new int[8];
#if JDK
        @NonNegative double d1 = Math.random() * a.Length;
        @NonNegative int deref = (int) (Math.random() * a.Length);
#endif
        int deref2 = (int) (rand.NextDouble() * a.Length);
        if(TestHelper.nondet()) Contract.Assert(deref2 >= 0);
        int deref3 = rand.Next(a.Length);
        if(TestHelper.nondet()) Contract.Assert(deref3 >= 0);
    }
}
