using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class Index118 {

    public static void foo(String [] args) {
        Contract.Requires(args.Length == 4);
        for (int i = 1; i <= 3; i++) {
            int x = i;
            if(TestHelper.nondet()) Contract.Assert(x >= 1 && x <= 3);
            Console.WriteLine(args[i]);
        }
    }

    public static void bar(int i, String [] args) {
        Contract.Requires(i >= 0);
        Contract.Requires(args.Length == 4);
        if (i <= 3) {
            Console.WriteLine(args[i]);
        }
    }
}
